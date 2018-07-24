---
title: LSTM的PyTorch实现
categories: PyTorch
mathjax: true
date: 2018-06-19
---

基于PyTorch的LSTM实现。

<!-- more -->

PyTorch封装了很多常用的神经网络，要实现LSTM非常的容易。这里用官网的实例修改实现练习里面的character level LSTM。

首先还是老样子，import需要的module：
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)
```

然后为了将数据放到网络里面，我们需要做一个编码单词的函数：
```python
def prepare_char_sequence(word, to_ix):
    idxs = [to_ix[char] for char in word]
    return(torch.tensor(idxs, dtype=torch.long))


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    char_idxs = [prepare_char_sequence(w, char_to_ix) for w in seq]
    return torch.tensor(idxs, dtype=torch.long), char_idxs


training_data = [
    ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
    ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
]
word_to_ix = {}
char_to_ix = {}
for sent, tags in training_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
        for char in word:
            if char not in char_to_ix:
                char_to_ix[char] = len(char_to_ix)
print(word_to_ix)
print(char_to_ix)

tag_to_ix = {"DET": 0, "NN": 1, "V": 2}

# These will usually be more like 32 or 64 dimensional.
# We will keep them small, so we can see how the weights change as we train.
WORD_EMBEDDING_DIM = 5
CHAR_EMBEDDING_DIM = 5
CHAR_HIDDEN_DIM = 3
WORD_HIDDEN_DIM = 6
```

其实这里想想，如果是全文本，我们的character level编码也就26个字母表那么多。

然后我们定义一个character level的网络：
```python
class LSTMTagger(nn.Module):
    def __init__(self, char_embedding_dim, word_embedding_dim, char_hidden_dim, word_hidden_dim, char_size, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.char_embedding_dim = char_embedding_dim
        self.word_embedding_dim = word_embedding_dim
        self.char_hidden_dim = char_hidden_dim
        self.word_hidden_dim = word_hidden_dim
        self.char_embeddings = nn.Embedding(char_size, char_embedding_dim)
        self.word_embeddings = nn.Embedding(vocab_size, word_embedding_dim)

        self.char_lstm = nn.LSTM(char_embedding_dim, char_hidden_dim)
        self.word_lstm = nn.LSTM((word_embedding_dim + char_hidden_dim), word_hidden_dim)

        self.hidden2tag = nn.Linear(word_hidden_dim, tagset_size)
        self.char_hidden = self.init_hidden(self.char_hidden_dim)
        self.word_hidden = self.init_hidden(self.word_hidden_dim)

    def init_hidden(self, hidden_dim):
        return(torch.zeros(1, 1, hidden_dim),
               torch.zeros(1, 1, hidden_dim))

    def forward(self, sentence):
        char_lstm_result = []
        for word in sentence[1]:
            self.char_hidden = self.init_hidden(self.char_hidden_dim)
            char_embeds = self.char_embeddings(word)
            lstm_char_out, self.char_hidden = self.char_lstm(char_embeds.view(len(word), 1, -1), self.char_hidden)
            char_lstm_result.append(lstm_char_out[-1])

        word_embeds = self.word_embeddings(sentence[0])
        char_lstm_result = torch.stack(char_lstm_result)
        lstm_in = torch.cat((word_embeds.view(len(sentence[0]), 1, -1), char_lstm_result), 2)
        lstm_out, self.hidden = self.word_lstm(lstm_in, self.word_hidden)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence[0]), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores
```

在forward部分可以看到，这里有两个LSTM。第一个LSTM做的事情是将character拼成word，相当于是返回了一个character level的word embedding。然后用这个embedding和直接embedding的word vector拼到一起，放到第二个LSTM里面训练词性标注。另外要注意的是，这里虽然有两个LSTM模型，但是我们并没有定义第一个LSTM的loss function。因为我们要让这个网络按照最后词性标注的效果来训练，因此我们不需要定义这个网络的loss function。

定义一下相关的参数：

```python
model = LSTMTagger(CHAR_EMBEDDING_DIM, WORD_EMBEDDING_DIM, CHAR_HIDDEN_DIM, WORD_HIDDEN_DIM, len(char_to_ix), len(word_to_ix), len(tag_to_ix))
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)
```

看一下训练前的输出结果是什么：
```python
with torch.no_grad():
    inputs = prepare_sequence(training_data[0][0], word_to_ix)
    tag_scores = model(inputs)
    print(tag_scores)
```

再看一下训练300轮之后的结果：
```python
for epoch in range(300):
    for sentence, tags in training_data:
        model.zero_grad()
        model.hidden = model.init_hidden(WORD_EMBEDDING_DIM)
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = prepare_char_sequence(tags, tag_to_ix)
        tag_scores = model(sentence_in)

        loss = loss_function(tag_scores, targets)
        loss.backward()
        optimizer.step()

with torch.no_grad():
    inputs = prepare_sequence(training_data[0][0], word_to_ix)
    tag_scores = model(inputs)
    print(tag_scores)
```

一般而言，character level的LSTM会比word level的更有效果。这里因为是一个toy级别的，看不出太显著的差别来。如果是海量数据，一般而言会有比较明显的效果。

另外，原来的example是单向的LSTM，这里顺便做一个双向的。其实双向的LSTM就是正向一个，反向再一个，所以hidden的部分是两倍。所以要修改的地方就是网络的定义，将单向LSTM的hidden乘以2就好了：

```python
class LSTMTagger(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)

        self.hidden2tag = nn.Linear(hidden_dim * 2, tagset_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return(torch.zeros(1 * 2, 1, self.hidden_dim),
               torch.zeros(1 * 2, 1, self.hidden_dim))

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, self.hidden = self.lstm(
            embeds.view(len(sentence), 1, -1), self.hidden)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores
```

这样就简单实现了一个toy级别的双向LSTM和character level的单向LSTM。
