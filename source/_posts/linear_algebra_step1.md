---
title: 线性代数 01
categories: 统计学习
mathjax: true
date: 2018-05-25
keywords: 线性代数,基本概念
---

线性代数基本概念。

<!-- more -->

# Vector

vector就是一组数字，有两种，一种是row vector，一种是column vector。一般而言，我们没有特殊声明，说一个vector就是一个column vector。

$$
\text{row vector: } \begin{bmatrix} 1 \ 2 \ 3 \end{bmatrix} \\
\text{column vector: } \begin{bmatrix}1 \\ 2 \\ 3 \end{bmatrix}
$$

通常来说，我们用小写加粗的字母表示一个向量，比如$\mathbf{v}$，或者$\vec{v}$，不过一般来说出于方便，基本上也用普通小写的字母表示。每一个向量的元素用相同字母带上下标表示，比如$v_i$。

向量常见运算如下：
$$
c \vec{v} = [cv_i] \\
\vec{a} + \vec{b} = [a_i + b_i]
$$

向量乘法其实可以看作是单维矩阵，所以放到后面矩阵部分。

# Matrix

matrix就是一组vector。一般我们用大写加粗字母表示，比如$\mathbf{M}$。每一个元素用相同字母带上下表表示，比如$m_{ij}$，其中$i$表示第$i$行，$j$表示第$j$列。同样处于方便，很多时候直接用大写的字母表示矩阵。我们会叫一个行列相同的矩阵是方阵（square matrix）。

矩阵的常见运算如下：
$$
c \mathbf{M} = [c m_{ij}] \\
\mathbf{A} + \mathbf{B} = [a_{ij} + b_{ij}]
$$
这里需要注意的是，只有两个矩阵的形状相同才可以做加减法。矩阵的加法运算是符合加法结合律和交换律的。

矩阵还有一个是transpose，也就是按照对角线互换元素。一般记做$A^{\top}$

后面讨论矩阵乘法。矩阵运算是机器学习或者深度学习最重要的事情，尤其是矩阵的乘法，求导。下面讨论矩阵的乘法。

这里以矩阵乘向量为例。假设有一个矩阵$A$和向量$x$，我们将这个记做$Ax$这里需要注意，$A$的列数必须跟$x$的行数一致。

具体看这个计算过程是这样的：
$$
\begin{align}
Ax & =
\begin{bmatrix}
a_{11} & a_{12} & \cdots & a_{1n} \\
a_{21} & a_{22} & \cdots & a_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
a_{m1} & a_{m2} & \cdots & a_{mn}
\end{bmatrix}
\times
\begin{bmatrix}
x_1 \\
x_2 \\
\vdots \\
x_n
\end{bmatrix} \\
& =
\begin{bmatrix}
a_{11} x_1 + a_{12} x_2 + \cdots + a_{1n} x_n \\
a_{21} x_1 + a_{22} x_2 + \cdots + a_{2n} x_n \\
\vdots \\
a_{m1} x_1 + a_{m2} x_2 + \cdots + a_{mn} x_n
\end{bmatrix} \\
& =
x_1 \begin{bmatrix} a_{11} \\ a_{21} \\ \vdots \\ a_{m1} \end{bmatrix} +
x_2 \begin{bmatrix} a_{12} \\ a_{22} \\ \vdots \\ a_{m2} \end{bmatrix} +
\cdots +
x_n \begin{bmatrix} a_{1n} \\ a_{2n} \\ \vdots \\ a_{mn} \end{bmatrix}
\end{align}
$$

这是从两个不同的角度来看待矩阵乘法，第一个是我们最习惯使用的，用左边矩阵的行，乘以右边矩阵的列。第二个是相对不那么常见的方法，用右边矩阵的行乘左边矩阵的列，然后再加起来。本质上是一样的，但是相对而言第一种方法我自己比较习惯。矩阵乘矩阵其实就是将右边的矩阵拆成一个个vector来乘，不赘述。矩阵乘法最后得到的结果形状是$A_{mp} B_{pn} = M_{mn}$，也就是左边的行数，右边的列数。

这里说的是矩阵叉乘矩阵的计算方法，还有一种是矩阵点乘矩阵，英文上前者是product，后者是element-wise product。一般我们是将点乘记成$A \odot B$，这里要求两个矩阵形状一致。

点乘的计算规则就很简单了，就是各个位置的元素相乘。计算方式是：
$$
A \odot B =
\begin{bmatrix}
a_{ij} b_{ij}
\end{bmatrix}
$$

最基本的矩阵运算就是这样。推荐一个参考书，[Matrix Cookbook](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf)，可以作为日常参考书用。
