---
title: 台大李宏毅机器学习 16
category: 统计学习
mathjax: true
date: 2017-10-23
---

RNN入门

<!-- more -->

RNN是一种比较复杂的网络结构，每一个layer还会利用上一个layer的一些信息。

比如说，我们要做slot filling的task。我们有两个句子，一个是“arrive Taipei on November 2nd”，另一个是“leave Taipei on November 2nd”。我们可以发现在第一个句子中，Taipei是destination，而第二个句子中Taipei是departure。如果我们不去考虑Taipei前一个词的话，Taipei的vector只有一个，那么同样的vector进来吐出的predict就是一致的。所以我们在做的时候就需要把前一个的结果存起来，在下一个词进来的时候用了参考。

所以这样我们就在neuron中设计一个大脑来存储这个值。所以网络长这样：

<img src=../../images/blog/ml105.png>

我们用这个网络来举例。假设我们每个weight都是1，每个activation都是linear的。那么我们现在有一个序列$(1, 1)，(1, 1)，(2, 2)$，那么一开始memory里面的值是$(a_1 = 0, a_2 = 0)$，现在将序列第一个值传入network，我们得到$(x_1 = 1, x_2 = 1)$，因为active function都是linear的，所以经过第一个hidden layer，我们输出的就是$1 \times a_1 + 1 \times a_2 + 1 \times x_1 + 1 \times x_2$，两个节点一致。所以第一个hidden layer得到$(2, 2)$，同时我们将$(2， 2)$保存起来，更新一下得到$(a_1 = 2, a_2 = 2)$，output layer是$(4, 4)$，所以得到第一个$(y_1=4, y_2=4)$。第二个input $(1, 1)$，同样计算一下，hidden layer得到的是$(6, 6)$和$(a_1 = 6, a_2 = 6)$，output layer是$(12, 12)$。同理第三个input最后得到的output是$(32, 32)$，所最后得到的三个output序列是$(4, 4), (12, 12), (32, 32)$。

那么我们可以想一下，如果现在的序列顺序变化一下，结果是否会不一致？如果现在的序列是$(1, 1), (2, 2), (1, 1)$，我们得到的是$(4, 4), (16, 16), (36, 36)$。结果发生了变化。所以RNN对序列是敏感的，这样的特性就表示，在slot filling的task里面，我们前面的arrive和leave将会影响后面接着的Taipei的结果。
