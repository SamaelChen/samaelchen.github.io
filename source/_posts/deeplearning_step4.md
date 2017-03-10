---
title: 深度前馈网络
category: 深度学习
mathjax: true
date: 2017-02-04
---

深度前馈网络，机器学习是个大坑，第5章用统计学习基础这个坑来填。

<!-- more -->

Deep feedforward networks also called *multilayer perceptrons*(MLPs)

1) XOR

Single perceptron cannot solve XOR problem("exclusive or"). The XOR function is an operation on two binary values, $x_1$ and $x_2$ looks like:
$$
0 \otimes 0 = 0 \\
0 \otimes 1 = 1 \\
1 \otimes 0 = 1 \\
1 \otimes 1 = 0
$$
We can specify our complete network as:
$$
f(\boldsymbol{x}; \boldsymbol{W}, \boldsymbol{c}, \boldsymbol{w}, b) = \boldsymbol{w}^{\top} \max\{0, \boldsymbol{W}^{\top} \boldsymbol{x} + \boldsymbol{c} \} + b.
$$

We can now specify a solution to the XOR problem. Let
$$
\boldsymbol{W} = \begin{bmatrix}
1 & 1 \\
1 & 1
\end{bmatrix}, \\
\boldsymbol{c} = \begin{bmatrix}
0 \\
-1
\end{bmatrix}, \\
\boldsymbol{w} = \begin{bmatrix}
1 \\
-2
\end{bmatrix}, \\
$$
and $b=0$.

Let $\boldsymbol{X}$ be the design matrix containing four points in the binary input space, with one example per row:
$$
\boldsymbol{X} = \begin{bmatrix}
0 & 0 \\
0 & 1 \\
1 & 0 \\
1 & 1
\end{bmatrix}.
$$

The first step in the neural network is to multiply the input matrix by the first layer's weight matrix:
$$
\boldsymbol{XW} = \begin{bmatrix}
0 & 0 \\
1 & 1 \\
1 & 1 \\
2 & 2
\end{bmatrix}.
$$

Next, we add the bias vector $\boldsymbol{c}$, to get
$$
\begin{bmatrix}
0 & -1 \\
1 & 0 \\
1 & 0 \\
2 & 1
\end{bmatrix}.
$$

In this space, all the examples lie along a line with slope 1.

Now looking back, the neural network looks like:
<img src=http://image18.poco.cn/mypoco/myphoto/20170308/10/18449013420170308104737064.png?490x412_130>

To finish computing the value of $\boldsymbol{h}$ for each example, we use the Rectified Linear Units function(ReLU) as active function:
$$
\begin{bmatrix}
0 & 0 \\
1 & 0 \\
1 & 0 \\
2 & 1
\end{bmatrix}
$$

We finish by multiplying by the weight vector $\boldsymbol{w}$:
$$
\begin{bmatrix}
0 \\
1 \\
1 \\
0
\end{bmatrix}.
$$

2) Gradient-Based Learning

An important aspect of the design of a deep neural network is the choice of the cost function. In most cases, our parametric model defines a distribution $p(\boldsymbol{y} | \boldsymbol{x}; \boldsymbol{\theta})$ and we use the principle of maximum likelihood. We use the cross-entropy between the training data and the model's predictions as the cost function.

In most cases, we use the maximum likelihood to train neural networks. This cost function is
$$
J(\boldsymbol{\theta}) = - \mathbb{E}_{\mathbf{x,y} \sim \hat{p}_{data}} \log p_{model}(\boldsymbol{y} | \boldsymbol{x}).
$$

An advantage of this approach of deriving the cost function from maximum likelihood is that it removes the burden of designing cost functions for each model. Specifying a model $p(\boldsymbol{y} | \boldsymbol{x})$ automatically determines a cost function $\log p(\boldsymbol{y} | \boldsymbol{x})$.
