---
title: 线性代数 03
categories: 统计学习
mathjax: true
date: 2018-06-08
---

Reduced row echelon form

<!-- more -->

在求解线性方程组的时候，RREF可以看做是原来矩阵的等价，而且更易于求解。定义RREF前，先定义leading entry。我们称每一行自左往右首个非零元素为leading entry。那么一个row echelon form matrix满足：

    1. 每一个非零行都在全零行的上方
    2. 下一行的leading entry严格比上一行的leading entry靠右
    3. 每一个包含leading entry的列，自leading entry往下都是0

而如果这个矩阵额外满足另外两个条件：

    4. 如果包含leading entry的列，除了leading entry之外，其余元素全为0。且只有leading entry一个非零元素
    5. leading entry是1

那么，在这样的情况下，我们叫这样的矩阵是RREF。

实际上，当我们完成高斯消元法后，我们得到的最后matrix就是RREF的。

那我们从RREF可以学到什么？

首选，不论我们如何进行行之间的变化，列之间的关系不会发生变化。比如：

$$
A = \begin{bmatrix}1 &2 &-1 &2 &1 &2 \\
                    -1 &-2 &1 &2 &3 &6 \\
                    2 &4 &-3 &2 &0 &3 \\
                    -3 &-6 &2 &0 & 3 &9 \end{bmatrix} \\
                    \ \\
R = \begin{bmatrix}1 &2 &0 &0 &-1 &-5 \\
                    0 &0 &1 &0 &0 &-3 \\
                    0 &0 &0 &1 &1 &2 \\
                    0 &0 &0 &0 &0 &0 \end{bmatrix}
$$

如果我们现在定义$R$是$A$的RREF，我们可以很直观看到矩阵$A$的第二列是第一列的两倍，而$R$也是这样的。其余的列之间的线性关系也一样得到了继承。

我们将每一列仅包含一个非零元素，且这个非零元素是1的列叫做pivot columns。然后我们可以很自然看到，pivot columns一定是independent的。

我们会发现，如果我们现在得到的是一个方阵，那么转成RREF以后，如果每一个column都是independent的，那么最后得到的一定是identity matrix。

如果现在是一个瘦长型的矩阵，也就是行数大于列数的矩阵，那么转成RREF一定是一部分为identity matrix，另一部分是zero matrix。且，identity matrix一定在zero matrix上方。

而现在如果是一个矮胖的矩阵，也就是列比行多的矩阵，那么转成RREF一定不会是linear independent的。

那么我们RREF的non-zero row数目跟上一篇的Rank有什么关系呢。实际上这两个数目是刚好相等的。这事情其实回想一下Rank的定义就可以了。进一步观察就会发现，其实也跟pivot columns一样多。

RREF其实大部分的内容也就这么多。
