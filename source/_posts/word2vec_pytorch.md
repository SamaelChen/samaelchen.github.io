---
title: word2vec的PyTorch实现
category: 深度学习
mathjax: true
date: 2018-06-04
---

Word Embedding现在是现在NLP的入门必备，这里简单实现一个CBOW的W2V。

<!-- more -->

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
