---
title: 线性代数 05
categories: 统计学习
mathjax: true
date: 2018-06-20
keywords: 线性代数,逆矩阵
---

逆矩阵

<!-- more -->

假设，现在有一个向量$v$，经过矩阵$A$的变换，再经过矩阵$B$的变换，向量不变。此外，先经过$B$再经过$A$，仍然得到$v$。在这种情况下，我们就可以说$A$和$B$互逆。

那么严格一点，如果$n \times n$的矩阵$A$是可逆的，那么会存在一个$n \times n$的矩阵$B$使得$AB = BA = I$。在这种情况下，我们称$B$是$A$的一个逆矩阵，可以记为$A^{-1}$。

所以有一个很明显的点，当一个矩阵不是方阵的时候，必定是没有逆矩阵的。不过实际上在矩阵分析里面，这样的矩阵也可以求逆，叫做伪逆。

此外，一个矩阵如果有逆矩阵，那么一定只有唯一的一个逆矩阵。这个其实很好证明：
$$
AB = BA = I，AC = CA = I \\
B = BI = B(AC) = (BA)C = IC = C
$$

那逆矩阵有什么用呢，用逆矩阵可以解之前的线性方程组，直接得到经过高斯消元法之后的结果。那实际上，逆矩阵求解对机器来说是没效率的，因为机器求逆矩阵就用了RREF。

现在考虑一下，如果我们对矩阵的乘做逆运算会怎么样。也就是$(AB)^{-1}$。那么我们可以得到的是$(AB)^{-1} = B^{-1}A^{-1}$，这里要求两个矩阵都是可逆的。由此我们可以推广，$(A_1 A_2 \cdots A_k)^{-1} = A_k^{-1} \cdots A_2^{-1} A_1^{-1}$。

那如果$A$是可逆的，$A^{\top}$是不是也是可逆呢？答案很明显是可逆的，可以求一下，$AA^{-1} = (AA^{-1})^{\top} = (A^{-1})^{\top} A^{\top} = I$，然后反过来$A^{-1}A = (A^{-1}A)^{\top} = A^{\top} (A^{-1})^{\top} = I$。所以我们就知道，$(A^{\top})^{-1} = (A^{-1})^{\top}$。

那么如何判断一个矩阵是不是可逆的？在Elementary Linear Algebra里面，提供了十几种判断依据：

>(a) $A$ is invertible.
(b) The reduced row echelon form of $A$ is $I_n$.
(c) The rank of $A$ equals $n$.
(d) The span of the columns of $A$ is $R_n$.
(e) The equation $Ax = b$ is consistent for every $b$ in $R_n$.
(f) The nullity of $A$ equals zero.
(g) The columns of $A$ are linearly independent.
(h) The only solution of $Ax = 0$ is $\mathbf{0}$.
(i) There exists an $n \times n$ matrix $B$ such that $BA = I_n$.
(j) There exists an $n \times n$ matrix $C$ such that $AC = I_n$.
(k) $A$ is a product of elementary matrices.

也就是说，当一个矩阵$A$是方阵时，如果$A$可逆，上述的条件都是等价的。

判断完是否可逆之后，如何计算逆矩阵呢？

这里引入一个概念，叫做elementary matrix，比如说$E = \begin{bmatrix} 1 &0 &0 \\ 0 &0 &1 \\ 0 &1 &0 \end{bmatrix}$。那这个变换矩阵的作用就是交换第二行和第三行。我们回顾一下，之前说的RREF其实做的事情就是多次的elementary matrix乘原来的矩阵。那么如果我们的RREF是单位矩阵，那么其实$A$的逆矩阵就是这么多elementary matrix的乘积，不过这里要注意的是elementary matrix $E$的顺序。

矩阵的逆大概就是这么多内容，简单一点判断，如果矩阵是方阵，且满秩，就可逆。计算方法就用高斯消元法的那一套来。
