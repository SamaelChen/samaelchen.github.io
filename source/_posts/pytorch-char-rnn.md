---
title: Char-RNN生成古诗
categories:
  - [PyTorch]
  - [深度学习]
mathjax: true
date: 2018-09-28
keywords: [pytorch, 词向量, LSTM, 循环神经网络, 文本生成]
---

尝试用char-RNN生成古诗，本来是想要尝试用来生成广告文案的，测试一波生成古诗的效果。嘛，虽然我对业务兴趣不大，不过这个模型居然把我硬盘跑挂了，也是醉。

<!-- more -->

其实Char-RNN来生成文本的逻辑非常简单，就是一个字一个字放进去，让RNN开始学，按照前面的字预测下面的字。所以就要想办法把文本揉成我们需要的格式。

比如说，我们现在有一句诗“床前明月光，疑是地上霜”。那么我们的输入就是“床前明月光”，那么我们的预测就是“前明月光，”，其实就是错位一位。

然后我们要考虑的是如何批量的把数据喂进去，这里参考了[gluon的教程](http://zh.gluon.ai/chapter_recurrent-neural-networks/lang-model-dataset.html)上面的一个操作，因为诗歌是有上下文联系的，如果我们用随机选取的话，很可能就会丢掉很多有用的信息，所以我们还要想办法将诗歌的这种连续性保留下来。

mxnet教程的方法是先将所有的文本串成一行。所有的换行符替换为空格，所以空格在这里起到了分段的作用，空格也就有了意义。然后我们因为我们要批量训练，所以先按照我们每批打算训练多少行文本，将这一个超长的文本截断成这样，然后按照我们一次想看多少个字的窗口扫描过去。代码实现上如下：
```python
def data_iter_consecutive(corpus_indices, batch_size, num_steps):
    corpus_indices = torch.tensor(corpus_indices)
    data_len = len(corpus_indices)
    batch_len = data_len // batch_size
    indices = corpus_indices[0: batch_size*batch_len].reshape((
        batch_size, batch_len))
    epoch_size = (batch_len - 1) // num_steps
    for i in range(epoch_size):
        i = i * num_steps
        X = indices[:, i: i + num_steps]
        Y = indices[:, i + 1: i + num_steps + 1]
        yield X, Y
```

这样有一个好处就是可以保持诗句的连续性，效果上大概是：

```python
# 所有诗句拼成一行
[1, 2, 3, 4, 5, 6, 7, 8, 9， 10， 11， 12]
# batch_size = 2, num_steps = 3
# batch 1
[[1, 2, 3], [7, 8, 9]]
# batch 2
[[4, 5, 6], [10, 11, 12]]
```

这样一来，一句诗[1, 2, 3, 4, 5, 6]就能在不同batch里面保持连贯性了。

然后就是很简单设计网络：

```python
class lyricNet(nn.Module):
    def __init__(self, hidden_dim, embed_dim, num_layers, weight,
                 num_labels, bidirectional, dropout=0.5, **kwargs):
        super(lyricNet, self).__init__(**kwargs)
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
#         self.embedding = nn.Embedding(num_labels, self.embed_dim)
        self.rnn = nn.GRU(input_size=self.embed_dim, hidden_size=self.hidden_dim,
                          num_layers=self.num_layers, bidirectional=self.bidirectional,
                          dropout=self.dropout)
        if self.bidirectional:
            self.decoder = nn.Linear(hidden_dim * 2, self.num_labels)
        else:
            self.decoder = nn.Linear(hidden_dim, self.num_labels)

    def forward(self, inputs, hidden=None):
        embeddings = self.embedding(inputs)
        states, hidden = self.rnn(embeddings.permute([1, 0, 2]), hidden)
        outputs = self.decoder(states.reshape((-1, states.shape[-1])))
        return(outputs, hidden)

    def init_hidden(self, num_layers, batch_size, hidden_dim, **kwargs):
        hidden = torch.zeros(num_layers, batch_size, hidden_dim)
        return hidden
```

这里我用的是很简单的one-hot做词向量，当然数据量大一点可以考虑pretrained的字向量。不过直观感受上用白话文训练的字向量应该效果不会太好吧。

接着就可以开始训练了：

```python
for epoch in range(num_epoch):
    start = time.time()
    num, total_loss = 0, 0
    data = data_iter_consecutive(corpus_indice, batch_size, 35)
    hidden = model.init_hidden(num_layers, batch_size, hidden_dim)
    for X, Y in data:
        num += 1
        hidden.detach_()
        if use_gpu:
            X = X.to(device)
            Y = Y.to(device)
            hidden = hidden.to(device)
        optimizer.zero_grad()
        output, hidden = model(X, hidden)
        l = loss_function(output, Y.t().reshape((-1,)))
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
    if (epoch % 10 == 0) or (epoch == (num_epoch - 1)):
        print('epoch %d/%d, loss %.4f, norm %.4f, time %.3fs, since %dh %dm %ds'
              %(epoch+1, num_epoch, total_loss / num, norm, end-start, h, m, s))
```

这里的训练过程需要注意两个点，一个是hidden的initial，因为我们想要保持句子的连续性，所以我们hidden的initial只要每个epoch的第一次initial一下就可以了，后面训练的过程中需要从计算图中拿掉。另外就是因为有梯度爆炸的问题，所以我们需要对梯度进行修剪。

最后一个是我自己最容易犯错的地方，死活记不住的就是RNN的输入输出每个dimension都代表了什么含义。原始的RNN接受的输入是(seq_len, batch_size, embedding_dimension)，输出的是(seq_len, batch_size, num_direction * hidden_dim)。所以我们习惯的batch在先的数据需要在这里做一个permute，将batch和seq做一下调换。然后就是我们做分类的时候，直接flatten成为一个长向量的时候，其实已经变成了[seq_len, seq_len, ...]这样的样子。简单理解就是本来我们都是横着看诗歌的，现在模型的输出是竖着输出的。所以我们后面算loss的时候，y也需要做一个转置再flatten。

具体的可以看我的这个[notebook]('https://github.com/SamaelChen/hexo-practice-code/blob/master/pytorch/text%20generater/generate%20poem.ipynb')。

接下来可能想试一下的是如果不用这种方法的话，是不是可以用padding的方法把句子长度统一再训练。

另外强势推荐[最全中华古诗词数据库](https://github.com/chinese-poetry/chinese-poetry)。数据非常非常全了。

后面如果要做到很好的效果可以做的方向一个是做韵脚的信息，还有就是平仄的信息也带进去。

anyway，想了一下，这样训练完的hidden是不是就包含了一个作者的文风信息？！
