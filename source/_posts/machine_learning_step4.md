---
title: 台大李宏毅机器学习 04
category: 统计学习
mathjax: true
date: 2017-08-31
---

分类算法
<!-- more -->

分类算法就是将数据喂给模型，模型输出这一笔数据属于哪一类。

不同于回归，分类模型的 Loss function 可以记为
$$
L(f) = \sum_n \delta(f(x^n) \ne \hat{y}^n)。
$$

这一个 Loss function 不可微分，不能使用梯度下降的方法解。解法可以有感知机或者 SVM 等。

这里介绍另一种方法

<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/ml014.png>

上述方法有一个问题，如果数据不出现在 training data 里，怎么计算在某个类别下，这个 sample 的概率？比如下图中，海龟是 training data 外的数据，那么取到海龟的概率是否记为0？

<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/ml015.png>

假设数据符合高纬正态分布，这里可以使用极大似然法。

<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/ml016.png>

那么要让上面的 Likelihood 最大，就是要找到
$\mu^*, \Sigma^*$得到$\arg \max_{\mu,\Sigma} L(\mu, \Sigma)$。由于正态分布的概率密度函数在均值处取得最大值，因此相应的均值、标准差可以很直观计算出来。

<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/ml017.png>

用上图的计算方法，我们并不会得到一个很好的结果。因为我们没有考虑两个 class 的协方差。所以我们将原来的公式改进为：

<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/ml018.png>

理论上，这边可以用任意概率分布函数，根据实际的数据情况来决定。
