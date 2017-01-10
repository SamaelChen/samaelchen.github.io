---
title: 深度学习线性代数基础
categories: 深度学习
---
深度学习入门线性代数基础部分

1) Scalars, Vectors, Matrices and Tensors

**Scalars**: A scalar is a single number.
**Vectors**: A vector is an array of numbers which arranged in order. We can denote it as $\boldsymbol{x} =\begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{bmatrix}$. If we wanna index a set of elements of a vector, we can define a set $S=\{1,3,6\}$, and write $\boldsymbol{x}_S$ to access $x_1,\ x_3,\ x_6$. And $\boldsymbol{x}_{-S}$ is the vector containing all the elements without $x_1,\ x_3,\ x_6$.
**Matrices**: A matrix is a 2-D array of numbers. Denoted as $\boldsymbol{A}$, and $\boldsymbol{A} \in \mathbb{R}^{m \times n}$ means the matrix $\boldsymbol{A}$ has $m$ rows and $n$ columns. We use $A_{i,j}$ to represent the element of $\boldsymbol{A}$. $\boldsymbol{A}_{i,:}$ known as the $i$-th row, likewise, $\boldsymbol{A}_{:,j}$ is the $j$-th column. If we need to index matrix-valued expression that not a single letter, we can use subscripts after the expression. Like $f(\boldsymbol{A})_{i,j}$ gives element $(i,j)$ of the matrix computed by applying the function $f$ to $\boldsymbol{A}$.
**Tensors**: In the general case, an array of numbers arranged on a regular grid with a variable number of axes known as a tensor. We denote a tensor $\mathbf{A}$ and the element is $A_{i,j,k}$.

One important operation is *transpose*. Denoted as: $\boldsymbol{A}^{\top}$ defined such that $(A^{\top})_{i,j}=A_{j,i}$. We can write a vector into $\boldsymbol{x} = [x_1, x_2, \dots, x_n]^{\top}$.

We can add matrices to each other, as long as they have the same shape. $\boldsymbol{C} = \boldsymbol{A}+\boldsymbol{B}$ where $C_{i,j} = A_{i,j} + B_{i,j}$.

We can also add a scalar or multiply a matrix by a scalar. $\boldsymbol{D} = a \cdot \boldsymbol{B} + c$ where $D_{i,j} = a \cdot B_{i,j} + c$.

In deep learning context, we can also add a vector to a matrix: $\boldsymbol{C}= \boldsymbol{A}+\boldsymbol{b}$ where $C_{i,j} = A_{i,j}+b_j$. The implicit copying of $\boldsymbol{b}$ to lots locations called *broadcasting*.

2) Multiplying Matrices and Vectors

$\boldsymbol{C}=\boldsymbol{AB}$ where $C_{i,j} = \sum \limits_k A_{i,k}B_{k,j}$.

The element-wise product or Hadamard product  denoted as $\boldsymbol{A} \odot \boldsymbol{B}$, where $C_{i,j} = A_{i, j} \cdot B_{i,j}$.

We can regard $\boldsymbol{C}=\boldsymbol{AB}$ as the Hadamard product of the $i$-th row of $\boldsymbol{A}$ with the $i$-th column of $\boldsymbol{B}$.

The followings are some properties of matrix product.

+ $\boldsymbol{A}(\boldsymbol{B} + \boldsymbol{C})=\boldsymbol{AB}+\boldsymbol{AC}$
+ $\boldsymbol{A}(\boldsymbol{BC}) = (\boldsymbol{AB})\boldsymbol{C}$
+ $\boldsymbol{x^{\top}y} = \boldsymbol{y^{\top}x}$
+ $(\boldsymbol{AB})^{\top} = \boldsymbol{B^{\top}A^{\top}}$

The $\boldsymbol{Ax}=\boldsymbol{b}$ is $$\begin{align}
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

The inverse matrix of $\boldsymbol{A}$ denoted as $\boldsymbol{A}^{-1}$, and defined as
$$
\boldsymbol{A}^{-1}\boldsymbol{A}=\boldsymbol{I}
$$

4) Linear Dependence and Span

The solution of $\boldsymbol{A}\boldsymbol{x} = \boldsymbol{b}$ is $\boldsymbol{x} = \boldsymbol{A}^{-1} \boldsymbol{b}$. If $\boldsymbol{A}^{-1}$ exists, the equation has exact one solution. But for some values of $\boldsymbol{b}$, there may exist infinite solutions or have no solutions. It's not possible to have more than one but less than infinite solutions; if $\boldsymbol{x}$ and $\boldsymbol{y}$ both are solutions, then $\boldsymbol{z} = \alpha \boldsymbol{x} + (1-\alpha)\boldsymbol{y}$ is also a solution for any real $\alpha$.

We can use the *linear combination* to analyze how many solutions the equation has. We write it as $\sum \limits_i c_i \boldsymbol{v}^{(i)}$.

The *span* of a set of vectors is the set of all points obtainable by linear combination of the original vectors.

If $\boldsymbol{Ax} = \boldsymbol{b}$ has a solution, $\boldsymbol{b}$ is the span of the columns of $\boldsymbol{A}$. This particular span is the *columns space* or the *range* of $\boldsymbol{A}$.

If $m=n$ and $\textrm{rank}(\boldsymbol{A})=n$, there must exist a solution, but if $\textrm{rank}(\boldsymbol{A}) < n$, there may exist infinitely many solutions. The $\textrm{rank}$ of $\boldsymbol{A}$ is the number of linearly independent columns.

No set of $m$-dimensional vectors can have more than $m$-mutually linearly independent columns, but a matrix with more than $m$ columns may have more than one such set.

That means we need a square matrix. A square matrix with linearly dependent columns is *singular*.

If $\boldsymbol{A}$ is not square or is square but singular, it can still be possible to solve the equation. But, we can not use matrix inversion to find the solution.

Well, it's possible to define a inverse that multiplied on the right: $\boldsymbol{AA}^{-1} = \boldsymbol{I}$. For square matrices, the left inverse and right inverse are equal.

5) Norms

We can use
$$
\| \boldsymbol{x} \|_p = \Big( \sum \limits_i |x_i|^p \Big)^{\frac{1}{p}}, \quad p \in \mathbb{R}, \  p \ge 1
$$
to measure the size of a vector, called it $L^p$ norm.

Norms are functions mapping vectors to non-negative values. A norm satisfies the following properties:
+ $f(\boldsymbol{x}) = 0 \Rightarrow \boldsymbol{x}=\boldsymbol{0}$
+ $f(\boldsymbol{x} + \boldsymbol{y}) \le f(\boldsymbol{x}) + f(\boldsymbol{y})$
+ $\forall \alpha \in \mathbb{R}, f(\alpha \boldsymbol{x}) = |\alpha|f(\boldsymbol{x})$.

The $L^2$ norm is also called as the *Euclidean norm* and often denoted as $\| \boldsymbol{x} \|$. It's also common using squared $L^2$ norm to measure the size of a vector. We can calculate squared $L^2$ norm as $\boldsymbol{x}^{\top}\boldsymbol{x}$.

The squared $L^2$ norm increases slowly near the origin. If we wanna to discriminate between elements that are exactly zero and elements that are small but nonzero. In these case, we turn to use $L^1$ norm. Every time an element of $\boldsymbol{x}$ moves away from $0$ by $\varepsilon$, the $L^1$ norm increases by $\varepsilon$. The $L^1$ norm is often used as a substitute for the number of nonzero entries. We can calculate it easily in R/Python.

The other norm commonly uses is the $L^{\infty}$ norm, also known as the *max norm*. It simplifies the absolute value of the element with the largest magnitude in the vector.
$$
\| \boldsymbol{x} \|_{\infty} = \max \limits_i |x_i|
$$

We also use *Frobenius norm* to measure the size of a matrix.
$$
\| \boldsymbol{A} \|_F = \sqrt{\sum \limits_{i,j} A_{i,j}^2}
$$

The dot product of two vectors can be rewritten in
$$
\boldsymbol{x}^{\top} \boldsymbol{y} = \| \boldsymbol{x}\|_2 \| \boldsymbol{y}\|_2 \cos \theta
$$
where $\theta$ is the angle between $\boldsymbol{x}$ and $\boldsymbol{y}$.

6) Special Kinds of Matrices and Vectors

The matrix's main diagonal contains nonzero entries and the others are zero is *diagonal* matrix. We can also use $\mathrm{diag}(\boldsymbol{v})$ to represent it, where $\boldsymbol{v}$ is the diagonal of matrix.
$\mathrm{diag}(\boldsymbol{v})\boldsymbol{x} = \boldsymbol{v} \odot \boldsymbol{x}, \mathrm{diag}(\boldsymbol{v})^{-1} = \mathrm{diag}([1/v_1, \dots, 1/v_n]^T)$.

Not all diagonal matrix should be square. The rectangular diagonal matrices do not have inverses. But when we calculate $\boldsymbol{Dx}$, if the rows of $\boldsymbol{D}$ greater than the columns of $\boldsymbol{D}$, will create some of zeros. And, some of the last elements will be discarding if the number columns is greater than the number of rows.

If the matrix $\boldsymbol{A} = \boldsymbol{A}^T$, we call the matrix $\boldsymbol{A}$ as *symmetric* matrix.

We called the vector whose $L^2$ norm equals to 1 as *unit vector*.

If two vectors are *orthogonal*, their dot product should equals to zero. $\boldsymbol{x}^T \boldsymbol{y} = 0$.

The *orthonormal matrix* is a square matrix whose rows are mutually orthogonal, and columns are mutually orthogonal:
$$
\boldsymbol{A}^T \boldsymbol{A}= \boldsymbol{A} \boldsymbol{A}^T \Rightarrow \boldsymbol{A}^{-1} = \boldsymbol{A}^T
$$

7) Eigendecomposition

We use *eigen-decomposition* to decompose the matrix into a set of eigenvectors and eigenvalues.

An eigenvector of a square matrix $\boldsymbol{A}$ is a non-zero vector $\boldsymbol{v}$ such that multiplication by $\boldsymbol{A}$ alters merely the scale of $\boldsymbol{v}$:
$$
\boldsymbol{Av} = \lambda \boldsymbol{v}.
$$

The scalar $\lambda$ is the *eigenvalue* corresponding to this eigenvector. There exists lots eigenvectors, we look for unit eigenvectors.

The eigen-decomposition is
$$
\boldsymbol{A} = \boldsymbol{V} \mathrm{diag}(\boldsymbol{\lambda}) \boldsymbol{V}^{-1}.
$$
Where $\boldsymbol{V} = [\boldsymbol{v}^{(1)}, \dots, \boldsymbol{v}^{(n)}], \  \boldsymbol{\lambda}=[\lambda_1, \dots, \lambda_n]^{\top}.$

We cannot decompose every matrix into eigenvalues and eigenvectors. But, we often need to decompose a specific class of matrices that have a simple decomposition. Specifically, every real symmetric matrix can be decomposed into an expression using real-valued eigenvectors and eigenvalues:
$$
\boldsymbol{A} = \boldsymbol{Q \Lambda Q}^{\top},
$$
where $\boldsymbol{Q}$ is an orthogonal matrix composed of eigenvectors of $\boldsymbol{A}$, and $\boldsymbol{\Lambda}$ is a diagonal matrix. Since $\boldsymbol{Q}$ is an orthogonal matrix, we can think of $\boldsymbol{A}$ as scaling space by $\lambda_i$ in direction $\boldsymbol{v}^{(i)}$. See the figure below
<img src=http://image18.poco.cn/mypoco/myphoto/20170109/23/18449013420170109231523094.png?1196x916_130>

While any real symmetric matrix $\boldsymbol{A}$ have an eigendecomposition, the eigendecomposition may not be unique. If any two or more eigenvectors share the same eigenvalue, then any set of orthogonal vectors lying in their span are also eigenvectors with that eigenvalue, and we could equivalently choose a $\boldsymbol{Q}$ using those eigenvectors instead. This can be proofed as below:
$$
\textrm{suppose } \lambda_1=\lambda_2=\lambda, \textrm{ and eigenvector } \boldsymbol{v}_1, \boldsymbol{v}_2, \textrm{ and thsy share the same eigenvalue } \lambda \\
\boldsymbol{Av}_1 = \lambda \boldsymbol{v}_1 \\
\boldsymbol{Av}_2 = \lambda \boldsymbol{v}_2 \\
\textrm{Assume } \boldsymbol{\alpha}_1=a_1\boldsymbol{v}_1 + b_1 \boldsymbol{v}_2 \textrm{ and } \boldsymbol{\alpha}_2=a_2\boldsymbol{v}_1 + b_2 \boldsymbol{v}_2. \textrm{ Let } \boldsymbol{\alpha}_1 \perp \boldsymbol{\alpha}_2. \\
\begin{align}
\boldsymbol{A \alpha}_1 &= \boldsymbol{A}(a_1 \boldsymbol{v}_1 + b_1 \boldsymbol{v}_2) \\
&= a_1\boldsymbol{A}\boldsymbol{v}_1 + b_1\boldsymbol{A}\boldsymbol{v}_2 \\
&= a_1 \lambda \boldsymbol{v}_1 + b_1 \lambda \boldsymbol{v}_2 \\
&= \lambda(a_1 \boldsymbol{v}_1 + b_1 \boldsymbol{v}_2) \\
&= \lambda \boldsymbol{\alpha}_1
\end{align}
$$

A matrix whose eigenvalues are all positive is called *positive definite*. A matrix whose eigenvalues are all positive or zero-valued is called *positive semidefinite*. If all eigenvalues are negative, the matrix is *negative definite*, and if all eigenvalues are negative or zero-valued, it is *negative semidefinite*. Positive semidefinite matrices guarantee that $\forall \boldsymbol{x}, \  \boldsymbol{x}^{\top} \boldsymbol{Ax} \ge 0$. Positive definite matrices gurarantee that $\boldsymbol{x}^{\top} \boldsymbol{Ax}=0 \Rightarrow \boldsymbol{x} = \boldsymbol{0}$.
