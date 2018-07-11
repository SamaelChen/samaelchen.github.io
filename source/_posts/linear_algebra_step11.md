---
title: 线性代数 11
category: 统计学习
mathjax: true
date: 2018-07-11
---

向量正交，如果从几何的角度来看，向量的正交可以看作是两个向量垂直。

<!-- more -->

首先，我们下一些定义。我们将向量的长度叫做norm，记做$\| v \| = \sqrt{v_1^2 + v_2^2 + \cdots + v_n^2}$。那么两个向量之间的距离我们用两个向量差的norm表示，记做$\| v - u \|$。

然后向量有两种乘积，一种是点乘，一种是叉乘。叉乘就看作是只有一列的矩阵，然后用矩阵的叉乘方法就好了。至于点乘，实际上也可以看作是叉乘。定义如下：
$$
v \cdot u = \sum_i^n v_i u_i = v^{\top} u
$$
这里再说明一下，默认向量是列向量。

现在进入正题，向量正交就是两个向量的点内积为0，也就是$u \cdot v = 0$。那么很自然就会知道，零向量与所有的向量正交。

那么向量的点内积有一些运算性质：
假设有向量$u，v$，矩阵$A$， 常数$c$

> 1. $u \cdot u = \| u \|^2$
2. $u \cdot u = 0$ if and only if $u = 0$
3. $u \cdot v = v \cdot u$
4. $u \cdot (v + w) = u \cdot v + u \cdot w$
5. $(v + w) \cdot u = v \cdot u + w \cdot u$
6. $cu \cdot v = u \cdot cv$
7. $\| cu \| = |c| \| u \|$
8. $Au \cdot v = (Au)^{\top} v = u^{\top}A^{\top}v = u \cdot A^{\top}v$
9. $\| u+v \| \le \|u\| + \|v\|$

如果我们现在有个向量集合，集合里所有的向量互相正交，那么我们就叫这个集合是orthogonal set。那么如果刚好这里的向量都是单位向量，这个集合就可以叫做orthonomal basis。

现在回过头来看，这样的一个集合有什么用呢？这个向量集合是不是非常像之前的坐标系。然后进一步来看，假设现在有一个集合$S = \{ v_1 \; v_2 \; \cdots \; v_n \}$是一个orthogonal basis，有一个向量$u$是这些向量的线性组合，也就是说$u = c_1 v_1 + c_2 v_2 + \cdots + c_n v_n$，那么如果我们要求$c_i$，其实非常简单就是$c_i = \frac{u \cdot v_i}{\| v_i \|^2}$。如果现在再从几何的角度来看，这个$c_i$其实就是$u$在$v_i$上投影的长度。

那如果现在随便给一个basis，$\{u_1 \; u_2 \; \cdots \; u_n \}$，现在要将这个basis变成orthogonal basis，要做的是：
$$
\begin{align}
v_1 & = u_1 \\
v_2 & = u_2 - \frac{u_2 \cdot v_1}{\|v_1\|^2}v_1 \\
v_3 & = u_3 - \frac{u_3 \cdot v_2}{\|v_2\|^2}v_2 - \frac{u_3 \cdot v_1}{\|v_1\|^2}v_1 \\
& \vdots \\
v_n & = u_n - \frac{u_n \cdot v_{n-1}}{\|v_{n-1}\|^2}v_{n-1} - \cdots - \frac{u_n \cdot v_1}{\|v_1\|^2}v_1
\end{align}
$$
