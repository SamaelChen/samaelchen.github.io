---
title: word2vec的PyTorch实现
categories: PyTorch
mathjax: true
date: 2018-08-02
---

Word Embedding现在是现在NLP的入门必备，这里简单实现一个CBOW的W2V。

2018-07-06更新一发用一篇小说来训练模型的脚本。

2018-08-02更新一发negative sampling版本。

<!-- more -->

# negtive sampling版本

**2018-08-02更新基于negative sampling方法的W2V**

翻了之前项亮实现的MXNet版本的NCE，看的不甚理解，感觉他写的那个是NEG的样子，然后还是自己写一个简单的negative sampling来做这个事情。关于NCE和NEG的区别，其实NEG就像是NCE的一个特殊情况，这个可以看[Notes on Noise Contrastive Estimation and Negative Sampling](https://arxiv.org/pdf/1410.8251.pdf)，或者是谷歌的一篇[总结](https://www.tensorflow.org/extras/candidate_sampling.pdf)。

关于negative sampling这里简单介绍一下，其实负采样的思路非常的简单，就是原来我们有多少个词，那么softmax就要算多少个词的概率，用负采样的方法就是将原来这样的巨量分类问题变成一个简单的二分类问题。也就是说，原来正确的label依然保留，接着只要sample出一小部分的负样本出来，然后做一个二分类问题就可以了。至于需要sample多少负样本，谷歌的C版本中是用了5个，好像哪里见过说不超过25个就可以了，但是现在忘了是哪篇文章了，可能不准确O__O "…

具体的公式推导其实很简单，可以看一下gluon关于[负采样的介绍](http://zh.gluon.ai/chapter_natural-language-processing/word2vec.html)。

所以实际上要实现这个负采样非常的容易，只要设计一个抽样分布，然后开始抽样就可以了。在很多词向量的资料里面都说到了，采样分布选用的是：
$$
P(w_i) = \frac{f(w_i)^{0.75}}{\sum(f(w_j)^{0.75})}
$$
这个其实非常像softmax，就是说用单个词的词频除以全部词频的和，原来的代码中加入了0.75的这个幂指数，完全是炼丹经验。

然后网上参考了一个开源的代码：

```python
class NEGLoss(nn.Module):
    def __init__(self, ix_to_word, word_freqs, num_negative_samples=5,):
        super(NEGLoss, self).__init__()
        self.num_negative_samples = num_negative_samples
        self.vocab_size = len(word_freqs)
        self.dist = F.normalize(torch.Tensor(
            [word_freqs[ix_to_word[i]] for i in range(self.vocab_size)]).pow(0.75), dim=0
        )

    def sample(self, num_samples, positives=[]):
        weights = torch.zeros((self.vocab_size, 1))
        for w in positives:
            weights[w] += 1.0
        for _ in range(num_samples):
            w = torch.multinomial(self.dist, 1)[0]
            while(w in positives):
                w = torch.multinomial(self.dist, 1)[0]
            weights[w] += 1.0
        return weights

    def forward(self, input, target):
        return F.nll_loss(input, target,
                          self.sample(self.num_negative_samples,
                                      positives=target.data.numpy()))
```

但是有个小问题就是，这里采用的其实是很取巧的一个方法，就是说，我每次会生成一个矩阵告诉pytorch究竟有哪6个sample被我拿到了，然后算negative log likelihood的时候就只算这6个。结果上来说，是实现了负采样，但是从算法效率上来说，其实并没有起到减少计算量的效果。

所以这里我们实现一个非常简单，类似nagative sampling，但是不是非常严格的采样函数：

```python
def neg_sample(num_samples, positives=[]):
    freqs_pow = torch.Tensor([freqs[ix_to_word[i]] for i in range(vocab_size)]).pow(0.75)
    dist = freqs_pow / freqs_pow.sum()
    w = np.random.choice(len(dist), (len(positives), num_samples), p=dist.numpy())
    if positives.is_cuda:
        return torch.tensor(w).to(device)
    else:
        return torch.tensor(w)
```

然后相应的，我们需要将我们的CBOW也变一下，按照$-\text{log} \frac{1}{1+\text{exp}\left(-\mathbf{u}_c^\top (\mathbf{v}_{o_1} + \ldots + \mathbf{v}_{o_{2m}}) /(2m)\right)}  - \sum_{k=1, w_k \sim \mathbb{P}(w)}^K \text{log} \frac{1}{1+\text{exp}\left((\mathbf{u}_{i_k}^\top (\mathbf{v}_{o_1} + \ldots + \mathbf{v}_{o_{2m}}) /(2m)\right)}$这个公式计算最后的loss。

```python
class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.embeddings.weight.data.uniform_(-0.5 / vocab_size, 0.5 / vocab_size)
    def forward(self, inputs, label):
        negs = neg_sample(5, label)
        u_embeds = self.embeddings(label).view(len(label), -1)
        v_embeds_pos = self.embeddings(inputs).mean(dim=1)
        v_embeds_neg = self.embeddings(negs).mean(dim=1)
        loss1 = torch.diag(torch.matmul(u_embeds, v_embeds_pos.transpose(0, 1)))
        loss2 = torch.diag(torch.matmul(u_embeds, v_embeds_neg.transpose(0, 1)))
        loss1 = -torch.log(1 / (1 + torch.exp(-loss1)))
        loss2 = -torch.log(1 / (1 + torch.exp(loss2)))
        loss = (loss1.mean() + loss2.mean())
        return(loss)
```

这里我将embedding层的权重进行了标准化，通过这样的标准化可以避免后面计算loss的时候出现无穷大的情况。然后其他参数不用做什么变化，开始训练看看效果。

```python
for epoch in range(100):
    total_loss = torch.Tensor([0])
    for context, target in tqdm(data_iter):
        context_ids = []
        for i in range(len(context[0])):
            context_ids.append(make_context_vector([context[j][i] for j in range(len(context))], word_to_ix))
        context_ids = torch.stack(context_ids)
        context_ids = context_ids.to(device)
        model.zero_grad()
        label = make_context_vector(target, word_to_ix)
        label = label.to(device)
        loss = model(context_ids, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    losses.append(total_loss)
    print('epoch %d loss %.4f' %(epoch, total_loss))
print(losses)
```

完整的notebook可以看[这个](https://github.com/SamaelChen/hexo-practice-code/blob/master/pytorch/langage%20model/w2v_ngs.ipynb)，效率上有质的提升。batchsize还是1024的时候大概压缩到15分钟左右，放到8192的时候大概一个epoch是10分钟。一本满足。

---

# toy 版本

首先import必要的模块：
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
```

CBOW的全称是continuous bag of words。和传统的N-gram相比，CBOW会同时左右各看一部分词。也就是说，根据左右两边的词，猜测中间的词是什么。而传统的N-gram是根据前面的词，猜后面的词是什么。在PyTorch的官网上给出了N-gram的实现。因此我们只需要在这个基础上进行简单的修改就可以得到基于CBOW的W2V模型。

```py
CONTEXT_SIZE = 2  # 2 words to the left, 2 to the right
raw_text = """We are about to study the idea of a computational process.
Computational processes are abstract beings that inhabit computers.
As they evolve, processes manipulate other abstract things called data.
The evolution of a process is directed by a pattern of rules
called a program. People create programs to direct processes. In effect,
we conjure the spirits of the computer with our spells.""".split()

# By deriving a set from `raw_text`, we deduplicate the array
vocab = set(raw_text)
vocab_size = len(vocab)

word_to_ix = {word: i for i, word in enumerate(vocab)}
data = []
for i in range(2, len(raw_text) - 2):
    context = [raw_text[i - 2], raw_text[i - 1],
               raw_text[i + 1], raw_text[i + 2]]
    target = raw_text[i]
    data.append((context, target))
print(data[:5])
```

首先定义我们需要的数据。这里的CBOW的Windows是2。因为单词没法直接拿来训练，因此这里我们用id来唯一标识每一个单词。然后我们需要做的一个事情就是将这些id编码成向量。14年谷歌放出来的C那一版我印象中是用的霍夫曼树再降维，现在的PyTorch和gluon都有embedding的类，可以将分类的数据直接编码成向量。所以我们现在用框架实现这个事情就非常简单了。

```py
class CBOW(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return(log_probs)
```

这个CBOW的类很简单，继承了PyTorch的Module类，然后第一步我们就做了一个embedding，然后做了一个隐藏层和一个输出层。最后我们做了一个softmax的动作来得到probability。这就是我们需要训练的神经网络。所以一直说W2V是一个单层的神经网络就是这个原因。

然后我们定义一个简单的函数，将单词转变成id
```py
def make_context_vector(context, word_to_ix):
    idxs = [word_to_ix[w] for w in context]
    return torch.tensor(idxs, dtype=torch.long)


make_context_vector(data[0][0], word_to_ix)
```

接着定义一些需要的参数：
```py
device = torch.device('cuda:0')
losses = []
loss_function = nn.NLLLoss()
model = CBOW(len(vocab), embedding_dim=10, context_size=CONTEXT_SIZE*2)
model.to(device)
optimizer = optim.SGD(model.parameters(), lr=0.1)
```

这里需要注意一点，context_size需要windows的大小乘2，因为CBOW同时左右都看了这些词，所以我们放进来的词实际上是windows乘2的数量。

这里我用了GPU来加速计算。如果没有GPU的可以注释掉所有跟device相关的代码，这个数据量不大，体会不到GPU的优势。

然后就是正式训练
```py
for epoch in range(10):
    total_loss = torch.Tensor([0])
    for context, target in data:
        context_ids = make_context_vector(context, word_to_ix)
        context_ids = context_ids.to(device)
        model.zero_grad()
        log_probs = model(context_ids)
        label = torch.tensor([word_to_ix[target]], dtype=torch.long)
        label = label.to(device)
        loss = loss_function(log_probs, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    losses.append(total_loss)
print(losses)
```

这样就是一个词向量的训练过程。如果我们需要得到embedding之后的结果，只需要将数据过一遍embedding这一层就可以了。

```py
model.embeddings(make_context_vector(data[0][0], word_to_ix))
```

可以对比一下训练前和训练后向量的差异。

---

# softmax低效率版本

2018-07-06更新内容：

之前写的那个是一个非常toy的网络，本质上就是了解一下word2vec是怎么一回事。不过完全不具备实操的能力。下面找了一些开源的语料，稍微修改了一下之前的脚本，还是基于CBOW的模型，这样就可以正常跑日常的数据。语料地址[https://github.com/lxrogers/CS221SAT/tree/master/data/Holmes_Training_Data](https://github.com/lxrogers/CS221SAT/tree/master/data/Holmes_Training_Data)。

先import一些必要的包，这里的tqdm是显示进度的。
```python
import torch
import torch.utils.data.dataloader as dataloader
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
import os
import re
import sys
import gc
from tqdm import tqdm
```

然后读入语料数据

```python
text = []
for file in os.listdir('Holmes_Training_Data/'):
    with open(os.path.join('Holmes_Training_Data', file), 'r', errors='ignore') as f:
        text.extend(f.read().splitlines())

text = [x.replace('*', '') for x in text]
text = [re.sub('[^ \fA-Za-z0-9_]', '', x) for x in text]
text = [x for x in text if x != '']
print(text[:10])
```

这里我ignore了一些文本读入的错误，然后过滤掉了符号。

因为语料是英文的，所以这里按照空格分割单词，比中文方便太多。

```python
raw_text = []
for x in text:
    raw_text.extend(x.split(' '))
raw_text = [x for x in raw_text if x != '']
```

分好词以后就可以开始构建词库

```python
vocab = set(raw_text)
vocab_size = len(vocab)
```

接着跟之前一样，构建一个提供训练数据的函数，并准备好训练数据

```python
def make_context_vector(context, word_to_ix):
    idxs = [word_to_ix[w] for w in context]
    return torch.tensor(idxs, dtype=torch.long)


word_to_ix = {word: i for i, word in enumerate(vocab)}
data = []
for i in range(2, len(raw_text) - 2):
    context = [raw_text[i - 2], raw_text[i - 1],
               raw_text[i + 1], raw_text[i + 2]]
    target = raw_text[i]
    data.append((context, target))
print(data[:5])
```

定义网络，这里要注意的是，因为数据比较大，我们是分batch喂进来的，因此之前forward的时候，我们把embedding的数据摊开的时候是摊成一行的，这里需要摊成每个batch_size的大小。

```python
class CBOW(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view(len(inputs), -1)
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return(log_probs)
```

定义各种参数

```python
CONTEXT_SIZE = 2
batch_size = 1024
device = torch.device('cuda:0')
losses = []
loss_function = nn.NLLLoss()
model = CBOW(vocab_size, embedding_dim=100,
             context_size=CONTEXT_SIZE*2)
model.to(device)
# model = torch.nn.DataParallel(model, device_ids=[0, 1]).cuda()
optimizer = optim.SGD(model.parameters(), lr=0.1)
```

我这里本来写了多卡的跑法，但是不知道是不是我写法有问题还是为什么，每次我跑第二块卡的时候，PyTorch都会去第一块卡开一块空间出来，就算我只是在第二块卡跑也会在第一块卡开一些空间。比较神奇，后面再研究一下。

然后定义一下data iterator。

```python
data_iter = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                        shuffle=False, num_workers=4)
```

然后这里要注意的是，shuffle参数会影响每次iter的速度，shuffle会慢很多。另外num_workers越多速度越快，但是很可能会内存爆炸，需要自己调一个合适的。

然后就可以开始训练了：

```python
for epoch in range(100):
    total_loss = torch.Tensor([0])
    for context, target in tqdm(data_iter):
        context_ids = []
        for i in range(len(context[0])):
            context_ids.append(make_context_vector([context[j][i] for j in range(len(context))], word_to_ix))
        context_ids = torch.stack(context_ids)
        context_ids = context_ids.to(device)
#         context_ids = torch.autograd.Variable(context_ids.cuda())
        model.zero_grad()
        log_probs = model(context_ids)
        label = make_context_vector(target, word_to_ix)
        label = label.to(device)
#         label = torch.autograd.Variable(label.cuda())
        loss = loss_function(log_probs, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    losses.append(total_loss)
    print('epoch %d loss %.4f' %(epoch, total_loss))
print(losses)
```

如果要多卡可以把to(device)的代码改成注释的代码就可以了。

然后就是需要**注意**的点了。

**这个网络的确是work的，训练完可以试一下发现queen-woman+man和king的cosine similarity的确比monkey或者其他的单词要高。但是这个网络的效率很低！很低！很低！（你觉得我会告诉你一个epoch需要跑一个半小时么）。**

原因在哪呢？其实很简单因为我这里使用的是softmax，也就是说，这个网络每一次训练都需要预测所有的词，比如我这个训练集里面有接近37万个词，那么每次就需要预测37万个类，效率之低可想而知。那么有什么解决方案呢？最早的时候，也就是谷歌C版本的解决方案是基于霍夫曼树的hierarchical softmax。后来DeepMind有一篇介绍把NCE（Noise-contrastive estimation）用来加速的论文[^1]。再后来又出现了negative sampling的论文[^2]。不过直观感受上，NCE和negative sampling是很像的，算是殊途同归吧。

后面过段时间更新对这两种方法的理解和代码。

[^1]: http://papers.nips.cc/paper/5165-learning-word-embeddings-efficiently-with-noise-contrastive-estimation.pdf
[^2]: http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf
