---
title: 台大李宏毅机器学习 05
category: 统计学习
mathjax: true
date: 2017-09-05
---

逻辑回归
<!-- more -->

逻辑回归是按照线性的方程进行分类的算法。原始的逻辑回归是针对二分类的。二分类的数据我们记取值范围为$[0, 1]$，由于回归方程不能直接对分类数据进行计算，因此我们引入$\sigma$函数。
$$
\sigma(z) = \frac{1}{1+\exp(-z)}.
$$
$\sigma$函数的作用就是将二分类的值平滑成一条曲线。
<img src=../../images/blog/ml019.png>

在开始逻辑回归之前，先回顾一下上一节课的内容。上一节课大致介绍了贝叶斯方法。贝叶斯方法是按照 posterior probability 来进行分类的。

posterior probability 在二分类时候写作：
$$
\begin{align}
\text{P}(C_1|x) &= \frac{\text{P}(x|C_1) \text{P}(C_1)}{\text{P}(x|C_1) \text{P}(C_1) + \text{P}(x|C_2) \text{P}(C_2)} \\
\end{align}
$$
