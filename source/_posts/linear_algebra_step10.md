---
title: 线性代数 10
categories: 统计学习
mathjax: true
date: 2018-07-10
---

在讲矩阵可对角化前，先引入一个概念，矩阵相似。如果存在方阵$A，B$，一个可逆矩阵$P$，使得$P^{-1} A P = B$，那么我们称$A$和$B$是相似的。那么如果现在$B$是一个对角矩阵的话，那么我们就称$A$是可对角化的（diagonalizable）。一般而言，这里会用$D$来表示对角矩阵。

<!-- more -->

那么对角化有什么意义呢？我们从公式出发看一下，将$P$表示为$[p_1 \; \cdots \; p_n]$，将$D$表示为$\begin{bmatrix} d_1 & \cdots & 0 \\ \vdots & \ddots & \vdots \\ 0 & \cdots & d_n \end{bmatrix}$。我们之前的公式是$P^{-1} A P = D$，所以$AP = PD$。

先看左边，$AP = [Ap_1 \; \cdots \; Ap_n]$，再看右边$PD = P[d_1 e_1 \; \cdots \; d_n e_n] = [P d_1 e_1 \; \cdots \; P d_n e_n] = [d_1 P e_1 \; \cdots \; d_n P e_n] = [d_1 p_1 \; \cdots \; d_n p_n]$。这不就是特征根么。

所以我们就看到$A$的特征向量可以组成一个向量空间$\mathbb{R}^n$。

那么如何对角化呢，只要找到n个线性无关的向量$p_i$，然后将这些向量组成一个矩阵，就可以得到可逆矩阵$P$。然后特征根只要按对角线排列就是$D$。

解法就是计算$\det(A - tI) = (t-\lambda_1)^{m_1} (t-\lambda_2)^{m_2} \cdots$。那么因为每个$\lambda$对应能有的eigenvector数量是小于等于指数$m$的，只要每一个指数$m$都等于eigenspace，那么我们就说$A$可以对角化。

比如矩阵$A = \begin{bmatrix} -1 & 0 & 0 \\ 0 & 1 & 2 \\ 0 & 2 & 1 \end{bmatrix}$，那么$A$的因式分解是$-(t+1)^2 (t-3)$。所以特征根是3和-1。而对应的特征向量就是$\begin{bmatrix}0 \\ 1 \\ 1 \end{bmatrix} \; \begin{bmatrix}1 \\ 0 \\ 0 \end{bmatrix} \; \begin{bmatrix}0 \\ 1 \\ -1 \end{bmatrix}$。这样我们就完成了对角化。

矩阵对角化的好处是如果要做连乘的时候，对角矩阵的连乘是非常简单的，这样就可以极大减少计算开销。也就是说$A^m = P^{-1} D^m P$。

最后其实回想一下之前的坐标系变换，对角化的过程其实就是一次坐标系的变换过程。
