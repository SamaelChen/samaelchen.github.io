---
title: 台大李宏毅机器学习 16
category: 统计学习
mathjax: true
date: 2017-10-19
---

RNN入门

<!-- more -->

RNN是一种比较复杂的网络结构，每一个layer还会利用上一个layer的一些信息。

比如说，我们要做slot filling的task。我们有两个句子，一个是“arrive Taipei on November 2nd”，另一个是“leave Taipei on November 2nd”。我们可以发现在第一个句子中，Taipei是destination，而第二个句子中Taipei是departure。如果我们不去考虑Taipei前一个词的话，Taipei的vector只有一个，那么同样的vector进来吐出的predict就是一致的。所以我们在做的时候就需要把前一个的结果存起来，在下一个词进来的时候用了参考。

所以这样我们就在neuron中设计一个大脑来存储这个值。所以网络长这样：

<img src=../../images/blog/ml105.png>

我们用这个网络来举例。假设我们每个weight都是1，每个activation都是linear的。那么我们现在有一个序列$(1, 1)，(1, 1)，(2, 2)$，那么一开始memory里面的值是$(a_1 = 0, a_2 = 0)$，现在
