---
title: 线性代数 08
categories: 统计学习
mathjax: true
date: 2018-07-02
keywords: 线性代数,坐标系
---

Coordinate system，就是坐标系。其实就是一组vector。

<!-- more -->

能够拿来做坐标系的vector set，很显然，按照上一篇博客里的内容，我们可以很自然想到，一个vector set必须符合两个条件：

1. 这个vector set是$\mathbb{R}^n$的span
2. 这个vector set里面的vector是independent的

那其实这个vector set就是basis。

如果现在我们的basis刚刚好每个vector都是相互垂直的单位向量，那么我们就会把这个坐标系叫做直角坐标系。

那么从这个角度来看，其实，我们就可以将矩阵乘法看作是坐标系转换。而且坐标系转换，矩阵一定是可逆的。

所以如果我们要做任意的坐标系和直角坐标系之间的转换，我们遵守如下的公式：

从$v_{B}$到$v$就是$v = B v_{B}$，反过来就是$v_{B} = B^{-1} v$。

事实上，坐标系转换，或者说线性变换是机器学习里面非常常见的一种情况。比如说PCA就是这样的一种变换。PCA有一点像是在找basis。另外如果了解NMF的话，NMF看上去更像是将一组数据的basis找出来。不过要注意的是，仅仅是看上去很像而已。

线性变换具有很显著的意义，将一个在原来坐标系下面很复杂的函数，通过线性变换以后就可能得到一个非常简单的函数。

如下图：

<img src='https://i.imgur.com/HJTQBeg.png'>

我们要做一个关于直线$y = \frac{1}{2} x$的映射关系。如果这个映射在直角坐标系下面，那么我们的变换矩阵是$\begin{bmatrix} 0.6 &0.8 \\ 0.8 &-0.6 \end{bmatrix}$。但是如果我们用这条直线作为横轴，垂直于这条直线的向量为纵轴，就会发现，其实在这个坐标系$\begin{bmatrix} 2 &-1 \\ 1 &2 \end{bmatrix}$内，变换只是$\begin{bmatrix} 1 &0 \\ 0 &-1 \end{bmatrix}$。
