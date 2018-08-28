---
title: 线性代数 13
categories: 统计学习
mathjax: true
date: 2018-07-17
keywords: [线性代数, SVD, pagerank]
---

两个线性代数里面在机器学习领域大放异彩的算法，SVD和PageRank。

<!-- more -->

之前对方阵有对角化分解，而且不是所有的方阵都可以对角化，但是SVD是所有矩阵都可以进行分解的。SVD分解的过程如下：

<img src='https://i.imgur.com/ZMo1wcb.png'>

这个公式会有什么特性呢？我们假设$U = \{u_1, u_2, \cdots, u_m\}$，$V = \{v_1, v_2, \cdots, v_n \}$，$\Sigma$是常数$\{\sigma_1, \sigma_2, \cdots, \sigma_k\}$的对角矩阵。这里有一个点要注意的就是$\Sigma$的样子大体上会是左上角一个对角矩阵，其余部分都是零的$m \times n$的矩阵。这些$\sigma$称为奇异值，这些奇异值会等于$A^{\top}A$的特征根的平方根。

那么如果矩阵经过SVD分解以后，一定会得到$Av_i = \begin{cases}\sigma_i u_i & \text{if } 1 \le i \le k \\ 0 & \text{if } i > k \end{cases}$，$A^{\top}u_i = \begin{cases}\sigma_i v_i & \text{if } 1 \le i \le k \\ 0 & \text{if } i > k \end{cases}$。

现在问题来了，如果给一个矩阵，要怎么计算奇异值？假设有一个矩阵$A = \begin{bmatrix} 0 & 1 & 2 \\ 1 & 0 & 1 \end{bmatrix}$，可以直观看到，这个矩阵是$3 \times 2$的矩阵，因此需要在$\mathbb{R}^3$和$\mathbb{R}^2$都要有orthogonal matrix。所以先构建矩阵$A^{\top}A = \begin{bmatrix}1 & 0 & 1 \\ 0 & 1 & 2 \\ 1 & 2 & 5 \end{bmatrix}$。那么做一个$\mathbb{R}^3$上面的orthogonal matrix，按照这个[博客](https://samaelchen.github.io/linear_algebra_step11/)最后的正交化方法，我们可以将矩阵正交化为$v_1 = \frac{1}{\sqrt{30}} \begin{bmatrix} 1 \\ 2 \\ 5 \end{bmatrix}$，$v_2 = \frac{1}{\sqrt{5}} \begin{bmatrix} 2 \\ -1 \\ 0 \end{bmatrix}$，$v_3 = \frac{1}{\sqrt{6}} \begin{bmatrix} 1 \\ 2 \\ -1 \end{bmatrix}$。然后求一下$A^{\top}A$的特征根分别是6，1和0。所以奇异值就是$\sqrt{6}$和1。然后就可以按照上面的公式算出$u_1 = \frac{1}{\sqrt{5}} \begin{bmatrix} 2 \\ 1 \end{bmatrix}$，$u_1 = \frac{1}{\sqrt{5}} \begin{bmatrix} -1 \\ 2 \end{bmatrix}$。

那么事实上算到这里，只要将上面的向量集合和奇异值排列好，就完成了矩阵$A$的SVD分解。也就是$$
A = \begin{bmatrix} \frac{2}{\sqrt{5}} & \frac{-1}{\sqrt{5}} \\ \frac{1}{\sqrt{5}} & \frac{2}{\sqrt{5}} \end{bmatrix} \begin{bmatrix} \sqrt{6} & 0 & 0 \\ 0 & 1 & 0 \end{bmatrix} \begin{bmatrix} \frac{1}{\sqrt{30}} & \frac{2}{\sqrt{5}} & \frac{1}{\sqrt{6}} \\ \frac{2}{\sqrt{30}} & \frac{-1}{\sqrt{5}} & \frac{2}{\sqrt{6}} \\ \frac{5}{\sqrt{30}} & 0 & \frac{-1}{\sqrt{6}} \end{bmatrix}^{\top}
$$

那么SVD跟PCA是有非常多相似的地方的，如果我们使用了全部的奇异值，那么我们就可以还原原来的矩阵，但是如果我们只取了前面的一部分奇异值，我们得到的就是一个损失了一部分信息的矩阵。SVD在机器学习的领域有非常多的应用，最常用的一个地方就是用在推荐算法里面，另外就是降维。此外，还有一个矩阵分解方法是NMF，解释性会更强一些。这个在之前机器学习的博客里面也有提到。

然后是PageRank。这个算法缔造了今天的谷歌，也被称作是最贵的eigen value。PageRank实际上是一个蛮复杂的模型，这里讲一个最简单的情况，后面找机会再认真学习一下。所以这里有一些矩阵分析里面的定理（虽然我也不是很懂）就直接记结论，证明过程以后再学吧。

首先我们假设这个世界上只有四个网页，他们的关系如下：

<img src='https://i.imgur.com/kpsOeB5.png'>

现在假设有一个人随机浏览网页，他到每一个网站的可能性都是一样的，那么根据上图的结果我们可以得到：
$$
\begin{align}
x_1 & = x_3 + \frac{1}{2} x_4 \\
x_2 & = \frac{1}{3} x_1 \\
x_3 & = \frac{1}{3} x_1 + \frac{1}{2} x_2 + \frac{1}{2} x_4 \\
x_4 & = \frac{1}{3} x_1 + \frac{1}{2} x_2
\end{align}
$$

所以我们就可以很简单得到这样一个矩阵$A = \begin{bmatrix}0 & 0 & 1 & \frac{1}{2} \\ \frac{1}{3} & 0 & 0 & 0 \\ \frac{1}{3} & \frac{1}{2} & 0 & \frac{1}{2} \\ \frac{1}{3} & \frac{1}{2} & 0 & 0\end{bmatrix}$。这样的矩阵我们叫做马尔科夫矩阵，或者叫转移矩阵。这种矩阵的特点是每一个行的和为1，或者每一个列的和为1。根据[Perron-Frobenius theorem](https://en.wikipedia.org/wiki/Perron%E2%80%93Frobenius_theorem)这样的矩阵必然有一个特征值为1。

所以PageRank实际上在做的事情就是计算$A$的特征根为1时候的特征向量。这个特征向量最后就是我们的网页排名。

如果想要对PageRank有多一点的了解可以上Wikipedia看一下[PageRank的页面](https://en.wikipedia.org/wiki/PageRank)，也可以直接看原来的[论文](http://ilpubs.stanford.edu:8090/422/1/1999-66.pdf)。另外有中文的这篇博客[http://blog.codinglabs.org/articles/intro-to-pagerank.html](http://blog.codinglabs.org/articles/intro-to-pagerank.html)，介绍比较全面，不过基本上没有数学证明过程。看看以后有没有空自己推导一遍，顺便Python实现一下。
