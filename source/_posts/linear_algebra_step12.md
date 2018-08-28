---
title: 线性代数 12
categories: 统计学习
mathjax: true
date: 2018-07-16
keywords: [线性代数,orthogonal matrix,symmetric matrix]
---

稍微介绍一下两种特殊的矩阵，orthogonal matrix和symmetric matrix。

<!-- more -->

orthogonal matrix其实就是矩阵里面每个向量相互独立的矩阵，如果是orthonormal的话，这些矩阵里的向量都是单位向量。比如说$\begin{bmatrix}\frac{1}{\sqrt{2}} & 0 & \frac{1}{\sqrt{2}} \\ \frac{1}{\sqrt{2}} & 0 & \frac{-1}{\sqrt{2}} \\
0 & 1 & 0 \end{bmatrix}$。

这样的矩阵有一些特性，首先，orthogonal matrix $Q$的transpose和inverse相等。也就是$Q^{\top} = Q^{-1}，且这两个矩阵都是orthogonal的$，另外，$\det(Q) = \pm 1$。最后，orthogonal matrix和orthogonal matrix叉乘之后还是orthogonal matrix。

orthogonal matrix还有一个很特殊的特性，就是向量和orthogonal matrix相乘以后，向量的norm不变。

另一种特殊矩阵是symmetric matrix，也就是类似$\begin{bmatrix}a & b \\ b & c \end{bmatrix}$。首先，symmetric matrix一定有实特征根。其次，symmetric matrix一定有orthogonal eigenvectors。最后，symmetric matrix一定是diagonalizable的。这里存在一个等价关系$A \text{ is symmetric等价于}  P^{\top}AP = D 或 A = PDP^{\top}$。而这的$P$包含$A$的特征向量，$D$是$A$的特征根组成的对角矩阵。
