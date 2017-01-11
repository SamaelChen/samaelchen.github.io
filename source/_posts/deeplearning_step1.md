---
title: 深度学习线性代数基础
categories: 深度学习
---
深度学习入门线性代数基础部分

1) Scalars, Vectors, Matrices and Tensors

**Scalars**: A scalar is a single number.
**Vectors**: A vector is an array of numbers which arranged in order. We can denote it as $\boldsymbol{x} =\begin{bmatrix} x\_1 \\\ x\_2 \\\ \vdots \\\ x\_n \end{bmatrix}$. If we wanna index a set of elements of a vector, we can define a set $S=\\{1,3,6\\}$, and write $\boldsymbol{x}\_S$ to access $x\_1,\ x\_3,\ x\_6$. And $\boldsymbol{x}\_{-S}$ is the vector containing all the elements without $x\_1,\ x\_3,\ x\_6$.
**Matrices**: A matrix is a 2-D array of numbers. Denoted as $\boldsymbol{A}$, and $\boldsymbol{A} \in \mathbb{R}^{m \times n}$ means the matrix $\boldsymbol{A}$ has $m$ rows and $n$ columns. We use $A\_{i,j}$ to represent the element of $\boldsymbol{A}$. $\boldsymbol{A}\_{i,:}$ known as the $i$-th row, likewise, $\boldsymbol{A}\_{:,j}$ is the $j$-th column. If we need to index matrix-valued expression that not a single letter, we can use subscripts after the expression. Like $f(\boldsymbol{A})\_{i,j}$ gives element $(i,j)$ of the matrix computed by applying the function $f$ to $\boldsymbol{A}$.
**Tensors**: In the general case, an array of numbers arranged on a regular grid with a variable number of axes known as a tensor. We denote a tensor $\mathbf{A}$ and the element is $A\_{i,j,k}$.

One important operation is *transpose*. Denoted as: $\boldsymbol{A}^{\top}$ defined such that $(A^{\top})\_{i,j}=A\_{j,i}$. We can write a vector into $\boldsymbol{x} = [x\_1, x\_2, \dots, x\_n]^{\top}$.

We can add matrices to each other, as long as they have the same shape. $\boldsymbol{C} = \boldsymbol{A}+\boldsymbol{B}$ where $C\_{i,j} = A\_{i,j} + B\_{i,j}$.

We can also add a scalar or multiply a matrix by a scalar. $\boldsymbol{D} = a \cdot \boldsymbol{B} + c$ where $D\_{i,j} = a \cdot B\_{i,j} + c$.

In deep learning context, we can also add a vector to a matrix: $\boldsymbol{C}= \boldsymbol{A}+\boldsymbol{b}$ where $C\_{i,j} = A\_{i,j}+b\_j$. The implicit copying of $\boldsymbol{b}$ to lots locations called *broadcasting*.

2) Multiplying Matrices and Vectors

$\boldsymbol{C}=\boldsymbol{AB}$ where $C\_{i,j} = \sum \limits\_k A\_{i,k}B\_{k,j}$.

The element-wise product or Hadamard product  denoted as $\boldsymbol{A} \odot \boldsymbol{B}$, where $C\_{i,j} = A\_{i, j} \cdot B\_{i,j}$.

We can regard $\boldsymbol{C}=\boldsymbol{AB}$ as the Hadamard product of the $i$-th row of $\boldsymbol{A}$ with the $i$-th column of $\boldsymbol{B}$.

The followings are some properties of matrix product.

+ $\boldsymbol{A}(\boldsymbol{B} + \boldsymbol{C})=\boldsymbol{AB}+\boldsymbol{AC}$
+ $\boldsymbol{A}(\boldsymbol{BC}) = (\boldsymbol{AB})\boldsymbol{C}$
+ $\boldsymbol{x^{\top}y} = \boldsymbol{y^{\top}x}$
+ $(\boldsymbol{AB})^{\top} = \boldsymbol{B^{\top}A^{\top}}$

The $\boldsymbol{Ax}=\boldsymbol{b}$ is $$\begin{align}
A\_{1,:} \boldsymbol{x} &= b\_1 \\\
A\_{2,:} \boldsymbol{x} &= b\_2 \\\
& \vdots \\\
A\_{m,:} \boldsymbol{x} &= b\_m ,
\end{align}$$
that is $$\begin{align}
A\_{1,1}x\_1 + A\_{1,2}x\_2 + &\cdots + A\_{1,n}x\_n = b\_1 \\\
A\_{2,1}x\_1 + A\_{2,2}x\_2 + &\cdots + A\_{2,n}x\_n = b\_2 \\\
&\vdots \\\
A\_{m,1}x\_1 + A\_{m,2}x\_2 + &\cdots + A\_{m,n}x\_n = b\_m
\end{align}$$

3) Identity and Inverse Matrices

We denote identity matrix as $\boldsymbol{I}$. All the diagonal elements are $1$, and all the others are zero.

The inverse matrix of $\boldsymbol{A}$ denoted as $\boldsymbol{A}^{-1}$, and defined as
$$
\boldsymbol{A}^{-1}\boldsymbol{A}=\boldsymbol{I}
$$

4) Linear Dependence and Span

The solution of $\boldsymbol{A}\boldsymbol{x} = \boldsymbol{b}$ is $\boldsymbol{x} = \boldsymbol{A}^{-1} \boldsymbol{b}$. If $\boldsymbol{A}^{-1}$ exists, the equation has exact one solution. But for some values of $\boldsymbol{b}$, there may exist infinite solutions or have no solutions. It's not possible to have more than one but less than infinite solutions; if $\boldsymbol{x}$ and $\boldsymbol{y}$ both are solutions, then $\boldsymbol{z} = \alpha \boldsymbol{x} + (1-\alpha)\boldsymbol{y}$ is also a solution for any real $\alpha$.

We can use the *linear combination* to analyze the exact numbers of solutions the equation has. We write it as $\sum \limits\_i c\_i \boldsymbol{v}^{(i)}$.

The *span* of a set of vectors is the set of all points obtainable by linear combination of the original vectors.

If $\boldsymbol{Ax} = \boldsymbol{b}$ has a solution, $\boldsymbol{b}$ is the span of the columns of $\boldsymbol{A}$. This particular span is the *columns space* or the *range* of $\boldsymbol{A}$.

If $m=n$ and $\textrm{rank}(\boldsymbol{A})=n$, there must exist a solution, but if $\textrm{rank}(\boldsymbol{A}) < n$, there may exist infinite solutions. The $\textrm{rank}$ of $\boldsymbol{A}$ is the number of linearly independent columns.

No set of $m$-dimensional vectors can have more than $m$-mutually linearly independent columns, but a matrix with more than $m$ columns may have more than one such set.

That means we need a square matrix. A square matrix with linearly dependent columns is *singular*.

If $\boldsymbol{A}$ is not square or is square but singular, it can still be possible to solve the equation. But, we can not use matrix inversion to find the solution.

Well, it's possible to define a inverse that multiplied on the right: $\boldsymbol{AA}^{-1} = \boldsymbol{I}$. For square matrices, the left inverse and right inverse are equal.

5) Norms

We can use
$$
\\| \boldsymbol{x} \\|\_p = \Big( \sum \limits\_i |x\_i|^p \Big)^{\frac{1}{p}}, \quad p \in \mathbb{R}, \  p \ge 1
$$
to measure the size of a vector, called it $L^p$ norm.

Norms are functions mapping vectors to non-negative values. A norm satisfies the following properties:
+ $f(\boldsymbol{x}) = 0 \Rightarrow \boldsymbol{x}=\boldsymbol{0}$
+ $f(\boldsymbol{x} + \boldsymbol{y}) \le f(\boldsymbol{x}) + f(\boldsymbol{y})$
+ $\forall \alpha \in \mathbb{R}, f(\alpha \boldsymbol{x}) = |\alpha|f(\boldsymbol{x})$.

The $L^2$ norm is also called as the *Euclidean norm* and often denoted as $\\| \boldsymbol{x} \\|$. It's also common using squared $L^2$ norm to measure the size of a vector. We can calculate squared $L^2$ norm as $\boldsymbol{x}^{\top}\boldsymbol{x}$.

The squared $L^2$ norm increases slow near the origin. If we wanna to discriminate between elements that are zero and elements that are small but nonzero. In these case, we turn to use $L^1$ norm. Every time an element of $\boldsymbol{x}$ moves away from $0$ by $\varepsilon$, the $L^1$ norm increases by $\varepsilon$. The $L^1$ norm is often used as a substitute for the number of nonzero entries. We can calculate it pretty easy in R/Python.

The other norm commonly uses is the $L^{\infty}$ norm, also known as the *max norm*. It simplifies the absolute value of the element with the largest value in the vector.
$$
\\| \boldsymbol{x} \\|\_{\infty} = \max \limits\_i |x\_i|
$$

We also use *Frobenius norm* to measure the size of a matrix.
$$
\\| \boldsymbol{A} \\|\_F = \sqrt{\sum \limits\_{i,j} A\_{i,j}^2}
$$

The dot product of two vectors can be rewritten in
$$
\boldsymbol{x}^{\top} \boldsymbol{y} = \\| \boldsymbol{x}\\|\_2 \\| \boldsymbol{y}\\|\_2 \cos \theta
$$
where $\theta$ is the angle between $\boldsymbol{x}$ and $\boldsymbol{y}$.

6) Special Kinds of Matrices and Vectors

The matrix's main diagonal contains nonzero entries and the others are zero is *diagonal* matrix. We can also use $\mathrm{diag}(\boldsymbol{v})$ to represent it, where $\boldsymbol{v}$ is the diagonal of matrix.
$\mathrm{diag}(\boldsymbol{v})\boldsymbol{x} = \boldsymbol{v} \odot \boldsymbol{x}, \mathrm{diag}(\boldsymbol{v})^{-1} = \mathrm{diag}([1/v\_1, \dots, 1/v\_n]^T)$.

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
Where $\boldsymbol{V} = [\boldsymbol{v}^{(1)}, \dots, \boldsymbol{v}^{(n)}], \  \boldsymbol{\lambda}=[\lambda\_1, \dots, \lambda\_n]^{\top}.$

We cannot decompose every matrix into eigenvalues and eigenvectors. But, we often need to decompose a specific class of matrices that have a simple decomposition. Specifically, we can decompose every real symmetric matrix into an expression using real-valued eigenvectors and eigenvalues:
$$
\boldsymbol{A} = \boldsymbol{Q \Lambda Q}^{\top},
$$
where $\boldsymbol{Q}$ is an orthogonal matrix composed of eigenvectors of $\boldsymbol{A}$, and $\boldsymbol{\Lambda}$ is a diagonal matrix. Since $\boldsymbol{Q}$ is an orthogonal matrix, we can think of $\boldsymbol{A}$ as scaling space by $\lambda\_i$ in direction $\boldsymbol{v}^{(i)}$. See the figure below
<img src=http://image18.poco.cn/mypoco/myphoto/20170109/23/18449013420170109231523094.png?1196x916\_130>

While any real symmetric matrix $\boldsymbol{A}$ have an eigendecomposition, the eigendecomposition may not be unique. If any two or more eigenvectors share the same eigenvalue, then any set of orthogonal vectors lying in their span are also eigenvectors with that eigenvalue, and we could equivalently choose a $\boldsymbol{Q}$ using those eigenvectors instead. We proof it as below:
$$
\textrm{suppose } \lambda\_1=\lambda\_2=\lambda, \textrm{ and eigenvector } \boldsymbol{v}\_1, \boldsymbol{v}\_2, \textrm{ and thsy share the same eigenvalue } \lambda \\\
\boldsymbol{Av}\_1 = \lambda \boldsymbol{v}\_1 \\\
\boldsymbol{Av}\_2 = \lambda \boldsymbol{v}\_2 \\\
\textrm{Assume } \boldsymbol{\alpha}\_1=a\_1\boldsymbol{v}\_1 + b\_1 \boldsymbol{v}\_2 \textrm{ and } \boldsymbol{\alpha}\_2=a\_2\boldsymbol{v}\_1 + b\_2 \boldsymbol{v}\_2. \textrm{ Let } \boldsymbol{\alpha}\_1 \perp \boldsymbol{\alpha}\_2. \\\
\begin{align}
\boldsymbol{A \alpha}\_1 &= \boldsymbol{A}(a\_1 \boldsymbol{v}\_1 + b\_1 \boldsymbol{v}\_2) \\\
&= a\_1\boldsymbol{A}\boldsymbol{v}\_1 + b\_1\boldsymbol{A}\boldsymbol{v}\_2 \\\
&= a\_1 \lambda \boldsymbol{v}\_1 + b\_1 \lambda \boldsymbol{v}\_2 \\\
&= \lambda(a\_1 \boldsymbol{v}\_1 + b\_1 \boldsymbol{v}\_2) \\\
&= \lambda \boldsymbol{\alpha}\_1
\end{align}
$$

A matrix whose eigenvalues are all positive is *positive definite*. A matrix whose eigenvalues are all positive or zero-valued is *positive semidefinite*. If all eigenvalues are negative, the matrix is *negative definite*, and if all eigenvalues are negative or zero-valued, it's *negative semidefinite*. Positive semidefinite matrices guarantee that $\forall \boldsymbol{x}, \  \boldsymbol{x}^{\top} \boldsymbol{Ax} \ge 0$. Positive definite matrices gurarantee that $\boldsymbol{x}^{\top} \boldsymbol{Ax}=0 \Rightarrow \boldsymbol{x} = \boldsymbol{0}$.

8) SVD

We can use eigenvectors and eigenvalues to decompose a matrix:
$$
\boldsymbol{A} = \boldsymbol{V}\mathrm{diag}(\boldsymbol{\lambda}) \boldsymbol{V}^{-1}
$$

The singular value decomposition provides another way to factorize a matrix:
$$
\boldsymbol{A} = \boldsymbol{UDV}^{\top}
$$

Suppose that $\boldsymbol{A}$ is an $m \times n$ matrix. Define $\boldsymbol{U}$ to be an $m \times m$ matrix, $\boldsymbol{D}$ as $m \times n$ matrix, and $\boldsymbol{V}$ as $n \times n$ matrix.

Note that, $\boldsymbol{D}$ is not necessarily square. If use R/Python, the dimension of $\boldsymbol{D}$ is $\min \\{m, n\\} \times \min \\{m, n\\}$. In R, it's $\boldsymbol{U}\_{m \times \min \\{m, n\\}}, \  \boldsymbol{D}\_{\min \\{m, n\\} \times \min \\{m, n\\}}, \  \boldsymbol{V}\_{n \times \min \\{m, n\\}}$. In Python, it's $\boldsymbol{U}\_{m \times m}, \  \boldsymbol{D}\_{\min \\{m, n\\} \times \min \\{m, n\\}}, \  \boldsymbol{V}\_{n \times n}.$

The elements along the diagonal of $\boldsymbol{D}$ are the *singular values* of the matrix $\boldsymbol{A}$. The columns of $\boldsymbol{U}$ are the *left-singular vectors*. The columns of $\boldsymbol{V}$ are the *right-singular vectors*.

The left-singular vectors of $\boldsymbol{A}$ are the eigenvectors of $\boldsymbol{AA}^{\top}$. The right-singular vectors of $\boldsymbol{A}$ are the eigenvectors of $\boldsymbol{A}^{\top} \boldsymbol{A}$. The non-zero singular vector of $\boldsymbol{A}$ are the square roots of the eigenvalues of $\boldsymbol{A}^{\top} \boldsymbol{A}$. The same is true for $\boldsymbol{AA}^{\top}$.

9) The Moore-Penrose Pseudoinverse

If a matrix is not square, it has no inversion matrix. Looking back at the equation $\boldsymbol{Ax}=\boldsymbol{y}.$

If $\boldsymbol{A}$ is taller than its wide, this equation may have no solutions. If $\boldsymbol{A}$ is wider than its tall, then it may have more than one solution.

We define *Moore-Penrose Pseudoinverse* as:
$$
\boldsymbol{A}^+ = \lim \limits\_{\alpha \rightarrow 0}(\boldsymbol{A}^{\top}\boldsymbol{A} + \alpha \boldsymbol{I})^{-1} \boldsymbol{A}^{\top}
$$

We don't use this definition in practices, but rather the formula:
$$
\boldsymbol{A}^+ = \boldsymbol{V} \boldsymbol{D}^+ \boldsymbol{U}^{\top},
$$
where $\boldsymbol{U}, \boldsymbol{D}, \boldsymbol{V}$ are the singular value decomposition of $\boldsymbol{A}$, and we can get $\boldsymbol{D}^+$ of a diagonal matrix $\boldsymbol{D}$ by taking the reciprocal of its non-zero elements then taking the transpose of the resulting matrix.

When $\boldsymbol{A}$ has more columns than rows, we choose the minimal $\\| \boldsymbol{x} \\|\_2$ among all possible solutions. When $\boldsymbol{A}$ has more rows than columns, using pseudoinverse gives us the $\boldsymbol{x}$ for which $\boldsymbol{Ax}$ is as close as possible to $\boldsymbol{y}$ based on $\\| \boldsymbol{Ax} - \boldsymbol{y} \\|\_2$

10) The Trace Operator

The trace operator gives us the sum of all the diagonal entries of a matrix:
$$
\mathrm{Tr}(\boldsymbol{A}) = \sum \limits\_i A\_{i,i}
$$

By using this operator, we can rewrite the Frobenius norm as:
$$
\\| \boldsymbol{A} \\|\_F = \sqrt{\mathrm{Tr(\boldsymbol{AA}^{\top})}}
$$

The trace operator is invariant to the transpose operator.
$$
\mathrm{Tr}(\boldsymbol{A}) = \mathrm{Tr}(\boldsymbol{A}^{\top})
$$

If the matrices allow the resulting product, we can define:
$$
\mathrm{Tr}(\boldsymbol{ABC}) = \mathrm{Tr}(\boldsymbol{CAB}) = \mathrm{Tr}(\boldsymbol{BCA}),
$$
more generally,
$$
\mathrm{Tr}(\prod \limits\_{i=1}^{n} \boldsymbol{F}^{(i)}) = \mathrm{Tr}(\boldsymbol{F}^{(i)} \prod \limits\_{i=1}^{n-1} \boldsymbol{F}^{(i)}).
$$

11) The Determinant

We denote determinant of a square matrix as $\det{A}$. It's a function mapping matrices to real scalars. The $\det{A}$ is equal to the product of all the eigenvalues of the matrix. It's a measure of how much multiplication by the matrix expends or contracts space.

12) PCA

The PCA is mapping a high dimension matrix to a low dimension space:
$$
f(\boldsymbol{x}) = \boldsymbol{c}
$$
and we can denote it:
$$
\boldsymbol{x} \approx g(f(\boldsymbol{x}))
$$

Let $g(\boldsymbol{x}) = \boldsymbol{Dc}$, PCA constrains the columns of $\boldsymbol{D}$ to be orthogonal to each other(unless $\boldsymbol{D}$ is square, it isn't technically "an orthogonal matrix"), and each columns has unit norm.

This algorithm works like that:
$$
\boldsymbol{c}^* = \underset{c}{\arg \min} \\| \boldsymbol{x} - g(\boldsymbol{c}) \\|\_2.
$$

We switch $L^2$ norm to the squared $L^2$ norm.
$$
\boldsymbol{c}^* = \underset{c}{\arg \min} \\| \boldsymbol{x} - g(\boldsymbol{c}) \\|\_2^2 \\\
$$

$$
\begin{align}
\\| \boldsymbol{x} - g(\boldsymbol{c}) \\|\_2^2 &= (\boldsymbol{x} - g(\boldsymbol{c}))^{\top} (\boldsymbol{x} = g(\boldsymbol{c})) \\\
&= \boldsymbol{x}^{\top} \boldsymbol{x} - \boldsymbol{x}^{\top}g(\boldsymbol{c}) - g(\boldsymbol{c})^{\top} \boldsymbol{x} + g(\boldsymbol{c})^{\top} g(\boldsymbol{c}) \\\
&= \boldsymbol{x}^{\top}\boldsymbol{x} - 2 \boldsymbol{x}^{\top} g(\boldsymbol{c}) + g(\boldsymbol{c})^{\top} g(\boldsymbol{c})
\end{align}\\\
$$

$$
\begin{align}
\boldsymbol{c}^* &= \underset{c}{\arg \min}(-2 \boldsymbol{x}^{\top}g(\boldsymbol{c}) + g(\boldsymbol{c})^{\top}g(\boldsymbol{c}))\\\
&= \underset{c}{\arg \min}(-2\boldsymbol{x}^{\top} \boldsymbol{Dc}+ \boldsymbol{c}^{\top} \boldsymbol{D}^{\top} \boldsymbol{Dc}) \\\
&= \underset{c}{\arg \min}(-2\boldsymbol{x}^{\top}\boldsymbol{Dc} + \boldsymbol{c}^{\top}\boldsymbol{c})
\end{align} \\\
\text{That is } \frac{\partial({-2\boldsymbol{x}^{\top}\boldsymbol{Dc} + \boldsymbol{c}^{\top}\boldsymbol{c})}}{\partial{\boldsymbol{c}}} = -2 \boldsymbol{D}^{\top} \boldsymbol{x} + 2 \boldsymbol{c} = \boldsymbol{0} \Rightarrow \boldsymbol{c} = \boldsymbol{D}^{\top}\boldsymbol{x}
$$

Define PCA reconstruction operation:
$$
r(\boldsymbol{x}) = g(f(\boldsymbol{x})) = \boldsymbol{D} \boldsymbol{D}^{\top} \boldsymbol{x}
$$

We can get:

$$
\boldsymbol{D}^* = \underset{D}{\arg \min} \sqrt{\sum \limits\_{i,j}(x\_j^{(i)}-r(x^{(i)})\_j)^2},\  \text{subject to } \boldsymbol{D}^{\top}\boldsymbol{D} = \boldsymbol{I}\_l \\\
$$

$$
(l \text{ is the number of components.}) \\\
\boldsymbol{D}^* = \underset{D}{\arg \min} \sum \limits\_i \\| \boldsymbol{x}^{(i)} - \boldsymbol{x}^{(i)} \boldsymbol{DD}^{\top} \\|\_2 = \underset{D}{\arg \min} \\| \boldsymbol{X} - \boldsymbol{XDD}^{\top} \\|\_F^2, \text{subject to } \boldsymbol{D}^{\top} \boldsymbol{D} = \boldsymbol{I}\_l
$$

$$
\begin{align}
\underset{D}{\arg \min} \\| \boldsymbol{X} - \boldsymbol{XDD}^{\top} \\|\_F^2 &=  \underset{D}{\arg \min} (\mathrm{Tr}[(\boldsymbol{X} - \boldsymbol{XDD}^{\top})^{\top}(\boldsymbol{X} - \boldsymbol{XDD}^{\top})]) \\\
&=\underset{D}{\arg \min} (\mathrm{Tr}(\boldsymbol{X}^{\top}\boldsymbol{X} -2\boldsymbol{X}^{\top}\boldsymbol{XDD}^{\top}+\boldsymbol{D}^{\top}\boldsymbol{DX}^{\top}\boldsymbol{XDD}^{\top})) \\\
&= \underset{D}{\arg \min}(-\mathrm{Tr}(\boldsymbol{X}^{\top}\boldsymbol{XDD}^{\top})) \\\
&= \underset{D}{\arg \max}(\mathrm{Tr}(\boldsymbol{X}^{\top}\boldsymbol{XDD}^{\top})), \text{subject to } \boldsymbol{D}^{\top} \boldsymbol{D} = \boldsymbol{I}\_l
\end{align}
$$
The $l$ eigenvectors corresponding to the largest eigenvalues is the matrix $\boldsymbol{D}$
