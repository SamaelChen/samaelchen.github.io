---
title: 台大李宏毅深度学习——RNN语言模型
categories: 深度学习
mathjax: true
date: 2018-06-01
keywords: 深度学习, RNN, 语言模型
---

RNN用于语言模型

<!-- more -->

在RNN出现以前，一般我们用的是N-gram model。所谓的N-gram model就是一个条件概率模型。比如2-gram用公式表示就是：
$$
P(w_1, w_2, w_3, \cdots, w_n) = P(w_1 | start)P(w_2 | w_1)\cdots P(w_n | w_{n-1})
$$
这边的概率就是计算$P(w_n | w_{n-1}) = \frac{\text{count}(w_{n-1} w_n)}{\text{count}(w_{n-1})}$。那相应的3-gram，4-gram就一样的道理。一般而言，n越大肯定效果就越好，但是计算量可想而知。

另外N-gram的问题就是，如果我们的语料库不够大的话，那么其实没有办法学到真正在语言空间中的概率。另外实操N-gram的时候，为了避免因为语料库太小导致一些条件概率变成0，我们一般会给一个非常非常小的概率，这个操作叫做smoothing。

另外还有一些smoothing的方法，比如说我们可以做matrix factorization。这个就跟我们平时推荐系统里面使用的矩阵分解一样，比如说SVD或者是NMF都可以。

那么RNN的好处就是，在参数量不增加的情况下，我们可以看得比N-gram更多。如下图：

<img src='https://i.imgur.com/gQgaden.png'>

呃，基于RNN的语言模型好像也就这些内容了。
