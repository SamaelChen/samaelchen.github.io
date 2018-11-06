---
title: 文本生成
categories: 深度学习
mathjax: true
date: 2018-11-02
keywords: [深度学习, NLP, text generation, 文本生成, LSTM]
---

最近诸事不顺，情绪不佳。继续做文本生成的事情。之前用的Char-RNN存在一定的缺陷，那就是你需要给定一个prefix，然后模型就会顺着prefix开始一个个往下预测。但是这样生成的文本随机性是很大的，所以我们希望能够让句子根据我们的关键词或者topic来生成。看了几篇论文，大框架上都是基于Attention的，其他的都是一些小的细节变化。这里打算实现两篇论文里的框架，一篇是哈工大的[Topic-to-Essay Generation with Neural Networks](http://ir.hit.edu.cn/~xcfeng/xiaocheng%20Feng's%20Homepage_files/final-topic-essay-generation.pdf)，另一篇是百度的[Chinese Poetry Generation with Planning based Neural Networks](https://arxiv.org/pdf/1610.09889.pdf)。

<!--more-->

第一篇论文里面放了三种策略，由简到繁分别是Topic-Averaged LSTM，Attention-based LSTM，以及Multi-Topic-Aware LSTM。

其实策略上来说，TAV-LSTM就是将topic的embedding做一个平均，然后作为prefix来训练，所以基本上网络设计上也和之前的Char-RNN差不多，比较容易实现。TAT-LSTM就是将topic做一个Attention，然后作为一个feature跟hidden并到一起喂到decoder里面去。MTA-LSTM还包含了一个叫做coverage vector的向量来计算topic的信息是否在训练过程中被喂进去了。

官方放了一个很久以前的TensorFlow版本的[MTA-LSTM](https://github.com/hit-computer/MTA-LSTM)，一方面我不喜欢TF，另一方面版本太老旧了，所以就用只能自己摸索着写PyTorch版本的了。数据就直接用的这个git上面提供的composition和zhihu两个数据。

然后这里都是用的贪婪法取候选词，没有做束搜索。当然，主要是因为懒，后面糟心事情过去了再说吧。

# TAV

TAV的大概工作原理上面也提到了，这里不赘述。然后同样偷懒，用了之前Char-RNN的模型直接修改。

首先常规套路，训练词向量。说起来，腾讯之前开源了一个800w+的词向量，也可以用用。这个就不多说了，很简单。

然后就是处理一下数据，首先我们要加入四个特殊字符PAD，BOS，EOS，和UNK。都是常规套路。

```python
fvec = KeyedVectors.load_word2vec_format('vec.txt', binary=False)
word_vec = fvec.vectors
vocab = ['<PAD>', '<BOS>', '<EOS>', '<UNK>']
vocab.extend(list(fvec.vocab.keys()))
word_vec = np.concatenate((np.array([[0]*word_vec.shape[1]] * 4), word_vec))
word_vec = torch.tensor(word_vec)
```

然后就是要做idx to word和word to idx的转换器。

```python
word_to_idx = {ch: i for i, ch in enumerate(vocab)}
idx_to_word = {i: ch for i, ch in enumerate(vocab)}
```

然后就是读数据，做iterator。

```python
essays = []
topics = []
with open('composition.txt', 'r') as f:
    for line in f:
        essay, topic = line.replace('\n', '').split(' </d> ')
        essays.append(essay.split(' '))
        topics.append(topic.split(' '))

corpus_indice = list(map(lambda x: [word_to_idx[w] for w in x], essays[:8000]))
topics_indice = list(map(lambda x: [word_to_idx[w] for w in x], topics[:8000]))
length = list(map(lambda x: len(x), corpus_indice))

def tav_data_iterator(corpus_indice, topics_indice, batch_size, num_steps):
    epoch_size = len(corpus_indice) // batch_size
    for i in range(epoch_size):
        raw_data = corpus_indice[i*batch_size: (i+1)*batch_size]
        key_words = topics_indice[i*batch_size: (i+1)*batch_size]
        data = np.zeros((len(raw_data), num_steps+1), dtype=np.int64)
        for i in range(batch_size):
            doc = raw_data[i]
            tmp = [1]
            tmp.extend(doc)
            tmp.extend([2])
            tmp = np.array(tmp, dtype=np.int64)
            _size = tmp.shape[0]
            data[i][:_size] = tmp
        key_words = np.array(key_words, dtype=np.int64)
        x = data[:, 0:num_steps]
        y = data[:, 1:]
        mask = np.float32(x != 0)
        x = torch.tensor(x)
        y = torch.tensor(y)
        mask = torch.tensor(mask)
        key_words = torch.tensor(key_words)
        yield(x, y, mask, key_words)
```

这里也是简单处理了，很多细节慢慢修改吧，然后就是这里的mask，我也是偷懒不去弄了，其实是标识那些词用来训练，哪些是padding的，我后面在loss function那里直接将PAD的权重改成0了。

然后就是定义网络。

```python
class TAVLSTM(nn.Module):
    def __init__(self, hidden_dim, embed_dim, num_layers, weight,
                 num_labels, bidirectional, dropout=0.5, **kwargs):
        super(TAVLSTM, self).__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_labels = num_labels
        self.bidirectional = bidirectional
        if num_layers <= 1:
            self.dropout = 0
        else:
            self.dropout = dropout
        self.embedding = nn.Embedding.from_pretrained(weight)
        self.embedding.weight.requires_grad = False
        self.rnn = nn.GRU(input_size=self.embed_dim, hidden_size=self.hidden_dim,
                          num_layers=self.num_layers, bidirectional=self.bidirectional,
                          dropout=self.dropout)
        if self.bidirectional:
            self.decoder = nn.Linear(hidden_dim * 2, self.num_labels)
        else:
            self.decoder = nn.Linear(hidden_dim, self.num_labels)

    def forward(self, inputs, topics, hidden=None):
        embeddings = self.embedding(inputs)
        topics_embed = self.embedding(topics)
        topics_embed = topics_embed.mean(dim=1)
        for i in range(embeddings.shape[0]):
            embeddings[i][0] = topics_embed[i]
        states, hidden = self.rnn(embeddings.permute([1, 0, 2]).float(), hidden)
        outputs = self.decoder(states.reshape((-1, states.shape[-1])))
        return(outputs, hidden)

    def init_hidden(self, num_layers, batch_size, hidden_dim, **kwargs):
        hidden = torch.zeros(num_layers, batch_size, hidden_dim)
        return hidden
```

基本结构没变化，就是forward的时候做了一点小修改，把第一个词变成topic average。

然后定义预测函数：

```python
def predict_rnn(topics, num_chars, model, device, idx_to_word, word_to_idx):
    output = [1]
    topics = [word_to_idx[x] for x in topics]
    topics = torch.tensor(topics)
    hidden = torch.zeros(num_layers, 1, hidden_dim)
    if use_gpu:
        hidden = hidden.to(device)
        topics = topics.to(device)
    for t in range(num_chars):
        X = torch.tensor(output).reshape((1, len(output)))
        if use_gpu:
            X = X.to(device)
        pred, hidden = model(X, topics, hidden)
        if pred.argmax(dim=1)[-1] == 2:
            break
        else:
            output.append(int(pred.argmax(dim=1)[-1]))
    return(''.join([idx_to_word[i] for i in output[1:]]))
```

设定一下参数：

```python
embedding_dim = 300
hidden_dim = 256
lr = 1e2
momentum = 0.0
num_epoch = 100
use_gpu = True
num_layers = 1
bidirectional = False
batch_size = 8
device = torch.device('cuda:0')
loss_function = nn.CrossEntropyLoss()

model = TAVLSTM(hidden_dim=hidden_dim, embed_dim=embedding_dim, num_layers=num_layers,
                num_labels=len(vocab), weight=word_vec, bidirectional=bidirectional)
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
if use_gpu:
    model.to(device)
```

接着训练就好了：

```python
since = time.time()
for epoch in range(num_epoch):
    start = time.time()
    num, total_loss = 0, 0
#     if epoch == 5000:
#         optimizer.param_groups[0]['lr'] = lr * 0.1
    data = tav_data_iterator(corpus_indice, topics_indice, batch_size, max(length)+1)
    hidden = model.init_hidden(num_layers, batch_size, hidden_dim)
    weight = torch.ones(len(vocab))
    weight[0] = 0
    for X, Y, mask, topics in tqdm(data):
        num += 1
        hidden.detach_()
        if use_gpu:
            X = X.to(device)
            Y = Y.to(device)
            mask = mask.to(device)
            topics = topics.to(device)
            hidden = hidden.to(device)
            weight = weight.to(device)
        optimizer.zero_grad()
        output, hidden = model(X, topics, hidden)
        l = F.cross_entropy(output, Y.t().reshape((-1,)), weight)
        l.backward()
        norm = nn.utils.clip_grad_norm_(model.parameters(), 1e-2)
        optimizer.step()
        total_loss += l.item()
    end = time.time()
    s = end - since
    h = math.floor(s / 3600)
    m = s - h * 3600
    m = math.floor(m / 60)
    s -= m * 60
    if(epoch % 10 == 0) or (epoch == (num_epoch - 1)):
        print('epoch %d/%d, loss %.4f, norm %.4f, time %.3fs, since %dh %dm %ds'
              %(epoch+1, num_epoch, total_loss / num, norm, end-start, h, m, s))
        print(predict_rnn(['妈妈', '希望', '长大', '孩子', '母爱'], 100, model, device, idx_to_word, word_to_idx))
```

具体还是看我的[notebook](https://github.com/SamaelChen/hexo-practice-code/blob/master/pytorch/text%20generater/Topic2Essay_train.ipynb)。不过梯度还是爆炸了，哎。

# TAT

就是在TAV的基础上修改，直接看[notebook](https://github.com/SamaelChen/hexo-practice-code/blob/master/pytorch/text%20generater/Topic2Essay_TAT.ipynb)吧。这个模型深刻地表达了我的内心正处在TAT的状态。

# 待续……心情真的是糟到极点
