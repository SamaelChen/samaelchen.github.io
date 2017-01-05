---
title: 深度学习线性代数基础
categories: 深度学习
---
深度学习入门线性代数基础部分

1) Scalars, Vectors, Matrices and Tensors

**Scalars**: A scalar is just a single number.
**Vectors**: A vector is an array of numbers which are arranged in order. We can denote it as $\boldsymbol{x} =\begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{bmatrix}$. If we wanna index a set of elements of a vector, we can define a set $S=\{1,3,6\}$, and write $\boldsymbol{x}_S$ to access $x_1,\ x_3,\ x_6$. And $\boldsymbol{x}_{-S}$ is the vector containing all of the elements without $x_1,\ x_3,\ x_6$.
**Matrices**: A matrix is a 2-D array of numbers. Denoted as $\boldsymbol{A}$, and $\boldsymbol{A} \in \mathbb{R}^{m \times n}$ means the matrix $\boldsymbol{A}$ has $m$ rows and $n$ columns. We use $A_{i,j}$ to represent the element of $\boldsymbol{A}$. $\boldsymbol{A}_{i,:}$ is known as the $i$-th row, likewise, $\boldsymbol{A}_{:,j}$ is the $j$-th column. If we need to index matrix-valued expression that not just a single letter, we can use subscripts after the expression. Like $f(\boldsymbol{A})_{i,j}$ gives element $(i,j)$ of the matrix computed by applying the function $f$ to $\boldsymbol{A}$.
**Tensors**: In the general case, an array of numbers arranged on a regular grid with a variable number of axes is known as a tensor. We denote a tensor $\mathbf{A}$ and the element is $A_{i,j,k}$.

One important operation is *transpose*. Denoted as: $\boldsymbol{A}^{\top}$. It is defined such that $(A^{\top})_{i,j}=A_{j,i}$. We can write a vector into $\boldsymbol{x} = [x_1, x_2, \dots, x_n]^{\top}$.

We can add matrices to each other, as long as they have the same shape. $\boldsymbol{C} = \boldsymbol{A}+\boldsymbol{B}$ where $C_{i,j} = A_{i,j} + B_{i,j}$.

We can also add a scalar or multiply a matrix by a scalar. $\boldsymbol{D} = a \cdot \boldsymbol{B} + c$ where $D_{i,j} = a \cdot B_{i,j} + c$.

In deep learning context, we can also add a vector to a matrix: $\boldsymbol{C}= \boldsymbol{A}+\boldsymbol{b}$ where $C_{i,j} = A_{i,j}+b_j$. The implicit copying of $\boldsymbol{b}$ to many locations is called *broadcasting*.

2) Multiplying Matrices and Vectors

$\boldsymbol{C}=\boldsymbol{AB}$ where $C_{i,j} = \sum \limits_k A_{i,k}B_{k,j}$.

The element-wise product or Hadamard product is denoted as $\boldsymbol{A} \odot \boldsymbol{B}$, where $C_{i,j} = A_{i, j} \cdot B_{i,j}$.

We can regard $\boldsymbol{C}=\boldsymbol{AB}$ as the Hadamard product of the $i$-th row of $\boldsymbol{A}$ with the $i$-th column of $\boldsymbol{B}$.

There are some properties of matrix product.

+ $\boldsymbol{A}(\boldsymbol{B} + \boldsymbol{C})=\boldsymbol{AB}+\boldsymbol{AC}$
+ $\boldsymbol{A}(\boldsymbol{BC}) = (\boldsymbol{AB})\boldsymbol{C}$
+ $\boldsymbol{x^{\top}y} = \boldsymbol{y^{\top}x}$
+ $(\boldsymbol{AB})^{\top} = \boldsymbol{B^{\top}A^{\top}}$

The equivalent of $\boldsymbol{Ax}=\boldsymbol{b}$ is $$\begin{align}
A_{1,:} \boldsymbol{x} &= b_1 \\
A_{2,:} \boldsymbol{x} &= b_2 \\
& \vdots \\
A_{m,:} \boldsymbol{x} &= b_m ,
\end{align}$$
that is $$\begin{align}
A_{1,1}x_1 + A_{1,2}x_2 + &\cdots + A_{1,n}x_n = b_1 \\
A_{2,1}x_1 + A_{2,2}x_2 + &\cdots + A_{2,n}x_n = b_2 \\
&\vdots \\
A_{m,1}x_1 + A_{m,2}x_2 + &\cdots + A_{m,n}x_n = b_m
\end{align}$$

3) Identity and Inverse Matrices

We denote identity matrix as $\boldsymbol{I}$. All the diagonal elements are $1$, and all the others are zero.

The inverse matrix of $\boldsymbol{A}$ is denoted as $\boldsymbol{A}^{-1}$, and it is defined as
$$
\boldsymbol{A}^{-1}\boldsymbol{A}=\boldsymbol{I}
$$
