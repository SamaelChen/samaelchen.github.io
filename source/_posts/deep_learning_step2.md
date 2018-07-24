---
title: 台大李宏毅深度学习——计算图模型
categories: 深度学习
mathjax: true
date: 2018-05-28
---

Computational Graph实际上是一种描述计算过程的“语言”。这种语言中用node表示variable，用edge表示operation。

<!-- more -->

举个简单的例子，比如有一个函数$y = f(g(h(x)))$，我们可以定义$u = h(x), v = g(u), y = f(v)$，这样我们就可以用计算图表示如下：

<img src='https://i.imgur.com/yxaoMlD.png'>

下面是一个具体的实例:
<img src='https://i.imgur.com/Bh3JsOn.png'>

从这个图我们可以计算，当$a = 2, b = 1$的时候，按照图的走向，我们可以算出$e=6$。

那么计算图的一个好处是我们可以比较简单实现梯度下降。如果现在我们要计算$\frac{\partial e}{\partial a} 和 \frac{\partial e}{\partial b}$，那么我们可以逆着图的方向，一步一步计算，首先计算$\frac{\partial e}{\partial c} = d = 2$，$\frac{\partial e}{\partial d} = c = 3$，然后我们发现，$a$只对$c$有影响，而$b$则同时对$c$和$b$有影响。那么我们顺着相反的路线就可以得到$\frac{\partial c}{\partial a} = 1$，$\frac{\partial c}{\partial b} = 1$，$\frac{\partial d}{\partial b} = 1$。这样我们很容易可以计算出两个偏微分分别是$\frac{\partial e}{\partial a} = \frac{\partial e}{\partial c} \frac{\partial c}{\partial a} = 2$，$\frac{\partial e}{\partial b} = \frac{\partial e}{\partial c} \frac{\partial c}{\partial b} + \frac{\partial e}{\partial d} \frac{\partial d}{\partial b}= 5$

那么如果现在碰到的是参数共享的计算图怎么办呢？例如下面的实例：
<img src='https://i.imgur.com/DdfMZRf.png'>

那么这时候我们需要先把每个$x$假装是完全不一样的变量计算。最后的时候再全部合并到一起。

认识了计算图之后，我们看如何计算神经网络的反馈。神经网络计算梯度下降分成两个步骤，一个是前馈，一个是反馈。公式上我们表示为：$\frac{\partial C}{\partial w_{ij}^l} = \frac{\partial z^l_i}{\partial w^l_{ij}} \frac{\partial C}{\partial z^l_i}$。

前半部分是前馈，将计算传递到最后；后半部分是反馈，将误差传递到前面。纯数学上的推导在之前的一篇[笔记](https://samaelchen.github.io/machine_learning_step6)中有介绍。这里讲一下如何利用计算图模型推导。

一个典型的前馈神经网络是这样的：

<img src='https://i.imgur.com/qjPbqyR.png'>

非常复杂的神经网络结构，用计算图表示很简洁。这里需要注意的是，对于任意一个神经网络，最后的cost只是一个scalar。但是实际上我们在计算的时候会发现一个事情，当我们计算$\frac{\partial z}{\partial a}$的时候，我们在计算的实际上是vector对vector的偏微分。那么应该怎么计算呢。这里介绍Jacobian Matrix。

比如我们现在有$y = f(x), x = \begin{bmatrix} x1 \\ x2 \\ x3 \end{bmatrix}, y = \begin{bmatrix} y1 \\ y2 \end{bmatrix}$。那么如果我们要求$\frac{\partial y}{\partial x}$，其实我们得到的就是$\begin{bmatrix} \partial y_1 / \partial x_1 &\partial y_1 / \partial x_2 &\partial y_1 / \partial x_3 \\
\partial y_2 / \partial x_1 &\partial y_2 / \partial x_2 &\partial y_2 / \partial x_3 \end{bmatrix}$这样的一个矩阵。这个矩阵我们就叫做是Jacobian Matrix。

首先我们算一下$\frac{\partial C}{\partial y}$，假设我们现在计算的是一个分类网络，那么我们得到的是：

<img src='https://i.imgur.com/8QJl3kA.png'>

因为这里我们用的是cross entropy：$C = -\log y_r$，所以我们可以知道当我们预测的$y_i$跟$\hat{y}_r$在$i=r$的时候有$\partial C / \partial y_r = -1 / y_r$，其余的位置因为真实值都是0，所以没有梯度。这一步还是比较好算的，我们得到的是一个很长的vector。

然后我们要计算的是$\frac{\partial y}{\partial z}$。因为这两个都是vector，所以很自然我们得到的是一个matrix：

<img src='https://i.imgur.com/TLlyVqM.png'>

这里有个点要注意的是，如果我们没有对$z$做softmax的操作，那么我们最后得到的一定是一个diagonal的matrix。此外，因为$z$到$y$只是做了一个activate function，所以也一定是相同维度的，所以必定会是一个方阵。

这里我们没有做softmax的情况下，同样只有在$y$和$z$下标一致的地方才有梯度。

接下去是比较棘手的地方，需要计算$\frac{\partial z}{\partial a}$和$\frac{\partial z}{\partial w}$。$\frac{\partial z}{\partial a}$还是比较好算的，因为这个计算好的结果刚好就是$W$，这个看公式就能看出来$z = \sum w_i a_i$，bias对$a$没有产生影响，所以这里不考虑。（备注：这边的PPT都是假设放进了一个矩阵$X$，行表示sample，列表示feature）

<img src='https://i.imgur.com/IANyU0v.png'>

相对难理解的是$\frac{\partial z}{\partial w}$。因为这里我们的计算是一个向量对一个矩阵的偏导数，最后得到的是一个三维的张量（tensor）。

<img src='https://i.imgur.com/U0TTb17.png'>

强行从二维的角度来看，其实就是每一个对角线上都是一个输入的样本矩阵$X$。这样一来，我们要做的事情就是一步步把计算的各个矩阵乘起来，就得到了梯度。

最后看一下在RNN里面如何做。RNN的基本结构是：

<img src='https://i.imgur.com/FVM31Gr.png'>

那么这里需要注意的是，因为我们每个RNN的block用的都是一样的function，所以实际上这些block是共享权重的，所以实际上我们要计算$\frac{\partial C}{\partial W^h}$在这个图里面需要计算三个，然后全部加起来：

<img src='https://i.imgur.com/5OuUGzn.png'>

现在基于计算图的框架比较多，MXNet的gluon，PyTorch都是。
