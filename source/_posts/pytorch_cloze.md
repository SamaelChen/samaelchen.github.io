---
title: 李宏毅深度学习作业——language model
categories:
  - [PyTorch]
  - [深度学习]
mathjax: true
date: 2018-08-28
keywords: [pytorch, word2vec, 词向量, LSTM, 循环神经网络, 完形填空]
---

之前用LSTM做过情感分析，李宏毅老师17年的课程第一个大作业是做一个完形填空的language model，试着做了一个简单的demo。
<!-- more -->

做完型填空其实很直观，就是跟CBOW很像，我们按照上下文猜被挖掉的那个词是什么。

这次用的还是之前训词向量的语料库，因为那个都是小说原文，所以我们要把数据揉成我们想要的形式，也就是context包含上下文，中间空掉的词是我们的target。

然后因为要训练LSTM，所以我们会再做一个padding的工作，最后看起来大概会是这样的：
```python
tensor([[[ 9405,  1236,  6282,   371,  1968,     0,     0,     0,     0,     0]],

        [[ 6085, 10586,   900,  7561,     0,     0,     0,     0,     0,     0]]])
```

形式上是$2 \times \text{batch_size} \times \text{seq_len}$。

网络的设置非常简单，前半部分过一个LSTM，后半部分过一个LSTM，然后将这两个网络的output拼到一起最后过一个fc。

这里因为有可能完形填空的时候空的是第一个词或者是最后一个词，所以我们会在句子开头和结尾加上<bos>和<eos>的标志。

一个示例可以看这个[notebook](‘https://github.com/SamaelChen/hexo-practice-code/blob/master/pytorch/langage%20model/LSTM-Full-text-Copy1.ipynb’)。

这个notebook的脚本没啥通用性，一个是其实没有解决unknown的词的问题，另外是没有解决训练效率的问题。PyTorch没有nce_loss或者是negative sampling这样的loss function，所以后面用softmax做cross entropy的时候复杂度是O(vocab_size)。之前写的negative sampling是针对word2vec写的，所以没什么通用性，看了其他人写的通用性的nce或者negative sampling，总感觉哪里怪怪的。后面还是要考虑自己实现一个。有点烦(╯﹏╰)。
