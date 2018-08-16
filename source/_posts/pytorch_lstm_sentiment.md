---
title: PyTorch实现LSTM情感分析
categories: PyTorch
mathjax: true
date: 2018-08-15
keywords: pytorch, word2vec, 词向量, LSTM, 循环神经网络, 情感分析
---

2018.08.16更新一个textCNN。

尝试使用LSTM做情感分析，这个gluon有非常详细的例子，可以直接参考gluon的[官方教程](http://zh.gluon.ai/chapter_natural-language-processing/sentiment-analysis.html)。这里尝试使用PyTorch复现一个。数据用的是IMDB的数据[http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz](http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz)

<!-- more -->

首先我们导入相关的package：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data.dataloader as dataloader
import torch.optim as optim
import torch.autograd as autograd
import torchtext.vocab as torchvocab
from torch.autograd import Variable
import tqdm
import os
import time
import re
import pandas as pd
import string
import gensim
import time
import random
import snowballstemmer
import collections
from collections import Counter
from nltk.corpus import stopwords
from itertools import chain
from sklearn.metrics import accuracy_score
```

然后我们定义读数的函数：

```python
def readIMDB(path, seg='train'):
    pos_or_neg = ['pos', 'neg']
    data = []
    for label in pos_or_neg:
        files = os.listdir(os.path.join(path, seg, label))
        for file in files:
            with open(os.path.join(path, seg, label, file), 'r', encoding='utf8') as rf:
                review = rf.read().replace('\n', '')
                if label == 'pos':
                    data.append([review, 1])
                elif label == 'neg':
                    data.append([review, 0])
    return data

train_data = readIMDB('aclImdb')
test_data = readIMDB('aclImdb', 'test')
```

接着是分词，这里只做非常简单的分词，也就是按照空格分词。当然按照一些传统的清洗方式效果会更好。

```python
def tokenizer(text):
    return [tok.lower() for tok in text.split(' ')]

train_tokenized = []
test_tokenized = []
for review, score in train_data:
    train_tokenized.append(tokenizer(review))
for review, score in test_data:
    test_tokenized.append(tokenizer(review))

vocab = set(chain(*train_tokenized))
vocab_size = len(vocab)
```

因为这个数据集非常小，所以如果我们用这个数据集做word embedding有可能过拟合，而且模型没有通用性，所以我们传入一个已经学好的word embedding。

```python

wvmodelwvmodel = gensim.models.KeyedVectors.load_word2vec_format('test_word.txt',
                                                          binary=False, encoding='utf-8')
```

这里的“test_word.txt”是我将glove的词向量转换后的结果，当时测试gensim的这个功能瞎起的名字，用的是glove的6B，100维的预训练数据。

然后一样要定义一个word to index的词典：

```python
word_to_idxword_to  = {word: i+1 for i, word in enumerate(vocab)}
word_to_idx['<unk>'] = 0
idx_to_word = {i+1: word for i, word in enumerate(vocab)}
idx_to_word[0] = '<unk>'
```

定义的目的是为了将预训练的weight跟我们的词库拼上。另外我们定义了一个unknown的词，也就是说没有出现在训练集里的词，我们都叫做unknown，词向量就定义为0。

然后就是编码：

```python
def encode_samples(tokenized_samples, vocab):
    features = []
    for sample in tokenized_samples:
        feature = []
        for token in sample:
            if token in word_to_idx:
                feature.append(word_to_idx[token])
            else:
                feature.append(0)
        features.append(feature)
    return features

def pad_samples(features, maxlen=500, PAD=0):
    padded_features = []
    for feature in features:
        if len(feature) >= maxlen:
            padded_feature = feature[:maxlen]
        else:
            padded_feature = feature
            while(len(padded_feature) < maxlen):
                padded_feature.append(PAD)
        padded_features.append(padded_feature)
    return padded_features
```

我们这里为了解决评论长度不一致的问题，将所有的评论都取500个词，超过的就取前500个，不足的补0。

整理一下训练数据：

```python
train_features = torch.tensor(pad_samples(encode_samples(train_tokenized, vocab)))
train_labels = torch.tensor([score for _, score in train_data])
test_features = torch.tensor(pad_samples(encode_samples(test_tokenized, vocab)))
test_labels = torch.tensor([score for _, score in test_data])
```

然后就是定义网络：

```python
class SentimentNet(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 bidirectional, weight, labels, use_gpu, **kwargs):
        super(SentimentNet, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.use_gpu = use_gpu
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding.from_pretrained(weight)
        self.embedding.weight.requires_grad = False
        self.encoder = nn.LSTM(input_size=embed_size, hidden_size=self.num_hiddens,
                               num_layers=num_layers, bidirectional=self.bidirectional,
                               dropout=0)
        if self.bidirectional:
            self.decoder = nn.Linear(num_hiddens * 4, labels)
        else:
            self.decoder = nn.Linear(num_hiddens * 2, labels)

    def forward(self, inputs):
        embeddings = self.embedding(inputs)
        states, hidden = self.encoder(embeddings.permute([1, 0, 2]))
        encoding = torch.cat([states[0], states[-1]], dim=1)
        outputs = self.decoder(encoding)
        return outputs
```

那这里需要注意几个点，第一，LSTM可以不initialize hidden，如果不initialize的话，那么PyTorch会默认初始为0。

另外就是LSTM这里传进去的数据格式是[seq_len, batch_size, embedded_size]。而我们传进去的数据是[batch_size, seq_len]的样子，那经过embedding之后的结果是[batch_size, seq_len, embedded_size]。所以我们这里要将第二个维度和第一个维度做个调换。而LSTM这边output的dimension和inputs是一致的，如果这里我们不做维度的调换，可以将LSTM的batch_first参数设置为True。然后我们要拿到每个batch的初始状态和最后状态还是一样要去做一个第一第二维度的调换。这里非常的绕，我在这里卡了好久(=@__@=)

第三就是我这里用了最初始的状态和最后的状态拼起来作为分类的输入。

另外有一点吐槽的就是，MXNet的dense层比较强大啊，不用定义输入的维度，只要定义输出的维度就可以了，操作比较骚啊。

然后我们把weight导进来：

```python
weight = torch.zeros(vocab_size+1, embed_size)

for i in range(len(wvmodel.index2word)):
    try:
        index = word_to_idx[wvmodel.index2word[i]]
    except:
        continue
    weight[index, :] = torch.from_numpy(wvmodel.get_vector(
        idx_to_word[word_to_idx[wvmodel.index2word[i]]]))
```

这里我们将不在glove里面的词全部填为0，后面想了一下，其实也可以试试这些全部随机试试。

接着定义参数就可以训练了。

```python
num_epochs = 5
embed_size = 100
num_hiddens = 100
num_layers = 2
bidirectional = True
batch_size = 64
labels = 2
lr = 0.8
device = torch.device('cuda:0')
use_gpu = True

net = SentimentNet(vocab_size=(vocab_size+1), embed_size=embed_size,
                   num_hiddens=num_hiddens, num_layers=num_layers,
                   bidirectional=bidirectional, weight=weight,
                   labels=labels, use_gpu=use_gpu)
net.to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=lr)
```

```python
train_set = torch.utils.data.TensorDataset(train_features, train_labels)
test_set = torch.utils.data.TensorDataset(test_features, test_labels)

train_iter = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                         shuffle=True)
test_iter = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                        shuffle=False)
```

这个位置需要注意的是，我们在train加了一个shuffle，如果不加shuffle的话，模型会学到奇奇怪怪的地方去。

最后训练一下就好了

```python
for epoch in range(num_epochs):
    start = time.time()
    train_loss, test_losses = 0, 0
    train_acc, test_acc = 0, 0
    n, m = 0, 0
    for feature, label in train_iter:
        n += 1
        net.zero_grad()
        feature = Variable(feature.cuda())
        label = Variable(label.cuda())
        score = net(feature)
        loss = loss_function(score, label)
        loss.backward()
        optimizer.step()
        train_acc += accuracy_score(torch.argmax(score.cpu().data,
                                                 dim=1), label.cpu())
        train_loss += loss
    with torch.no_grad():
        for test_feature, test_label in test_iter:
            m += 1
            test_feature = test_feature.cuda()
            test_label = test_label.cuda()
            test_score = net(test_feature)
            test_loss = loss_function(test_score, test_label)
            test_acc += accuracy_score(torch.argmax(test_score.cpu().data,
                                                    dim=1), test_label.cpu())
            test_losses += test_loss
    end = time.time()
    runtime = end - start
    print('epoch: %d, train loss: %.4f, train acc: %.2f, test loss: %.4f, test acc: %.2f, time: %.2f' %
          (epoch, train_loss.data / n, train_acc / n, test_losses.data / m, test_acc / m, runtime))
```

也可以直接看我的[notebook](https://github.com/SamaelChen/hexo-practice-code/blob/master/pytorch/langage%20model/lstm-sentiment.ipynb)

后面试试textCNN，感觉也挺骚气的。

---

2018.08.16 更新一个textCNN的玩法。

CNN太熟了，很容易搞，其实只要把网络改一下，其他的动都不用动：

```python
class textCNN(nn.Module):
    def __init__(self, vocab_size, embed_size, seq_len, labels, weight, **kwargs):
        super(textCNN, self).__init__(**kwargs)
        self.labels = labels
        self.embedding = nn.Embedding.from_pretrained(weight)
        self.embedding.weight.requires_grad = False
        self.conv1 = nn.Conv2d(1, 1, (3, embed_size))
        self.conv2 = nn.Conv2d(1, 1, (4, embed_size))
        self.conv3 = nn.Conv2d(1, 1, (5, embed_size))
        self.pool1 = nn.MaxPool2d((seq_len - 3 + 1, 1))
        self.pool2 = nn.MaxPool2d((seq_len - 4 + 1, 1))
        self.pool3 = nn.MaxPool2d((seq_len - 5 + 1, 1))
        self.linear = nn.Linear(3, labels)

    def forward(self, inputs):
        inputs = self.embedding(inputs).view(inputs.shape[0], 1, inputs.shape[1], -1)
        x1 = F.relu(self.conv1(inputs))
        x2 = F.relu(self.conv2(inputs))
        x3 = F.relu(self.conv3(inputs))

        x1 = self.pool1(x1)
        x2 = self.pool2(x2)
        x3 = self.pool3(x3)

        x = torch.cat((x1, x2, x3), -1)
        x = x.view(inputs.shape[0], 1, -1)

        x = self.linear(x)
        x = x.view(-1, self.labels)

        return(x)
```

这里的网络设计很简单，就是用三个filter去扫一遍文章，filter的尺寸其实就是我们一次看多少个词。这样扫完以后是三个向量，然后pooling一下得到三个实数。把这三个实数拼成一个向量，然后用fc分类一下就结束了。

然后初始化网络：

```python
net = textCNN(vocab_size=(vocab_size+1), embed_size=embed_size,
              seq_len=500, labels=labels, weight=weight)
```

其他的都没改，就可以直接跑了。速度上CNN比LSTM的参数少，速度快很多，不过只跑几轮的话效果差一点。
