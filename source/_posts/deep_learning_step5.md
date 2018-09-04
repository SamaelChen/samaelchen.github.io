---
title: 台大李宏毅深度学习——Batch normalization & SELU
categories: 深度学习
mathjax: true
date: 2018-09-03
keywords: [深度学习, batch normalization, SELU]
---

学习一下batch normalization和SELU，顺便看点深度学习的八卦。

<!-- more -->

Batch normalization是一个比较新的深度学习技巧，但是在深度学习的实作中有非常迅速成为中流砥柱。

normalization是以前统计学习比较常用的一种方法，因为对于损失函数而言，$L(y, \hat{y})$会受到输入数据的影响。这个其实是非常直观的，比如说一个数据有两个维度，一个维度都是1-10的范围内波动的，另一个维度是1000-10000之间波动的，那么如果$y=x_1 + x_2$很明显后一个维度的数据对$y$的影响非常大。

那么在这种情况下，我们做梯度下降，在scale大的维度上梯度就比较大，但是在scale小的地方梯度就比较小。这个在我之前学[梯度下降的博客](‘https://samaelchen.github.io/machine_learning_step3/’)里面也有。大概图形上看就是下面这样：

<img src='https://i.imgur.com/mb0vi91.png'>

那这样我们在不同维度上的梯度下降步长是不一样的。所以在统计学习或者传统的机器学习里面，为了加快收敛的速度，虽然用二阶导可以解决，但是一般用feature scaling就可以了。

而batch normalization其实也是使用了这样的理念。一般而言，我们做normalization就是$\frac{x-\mu}{\sigma}$，那batch normalization其实就是在每一个layer的input前做这么一下操作。

那batch normalization和normalization的差别其实就在于batch这个地方。我们知道平时我们训练深度学习网络的时候避免炸内存，会将数据分批导进去训练，在这种情况下，我们其实是没有办法得到全局的$\mu$和$\sigma$的。所以事实上，batch normalization每一次算的都是一个batch的$\mu \ \& \ \sigma$。

那整个流程看上去就是下图这样的：
<p align='center'>
<img src='https://i.loli.net/2018/09/04/5b8e3c359e425.png' width=70%>
</p>
那实际上可以将这个过程看作是一个hidden layer来处理。

如果说觉得这样全部normalization到0，1这样的形式可能有些activation function效果不好，所以我们可以考虑一下再加一层linear layer来转换一下，那流程上就是：
<p align='center'>
<img src='https://i.loli.net/2018/09/04/5b8e3e03c9485.png' width=70%>
</p>
当然，如果好巧不巧，机器学着学着，刚好$\beta$和$\gamma$跟前面的一样，那么这轮的batch normalization就白做了。不过一般来说不会这么巧。

那么在训练过程中，我们一般都是一个batch一个batch喂进去，但是test的时候，我们一般是一口气全部过模型一遍，那么我们并没有办法得到一个合适的$\mu$和$\sigma$。那么一种解决方法是计算一下全部training set的均值和标准差，另一种方法是，每次训练后，我们都保留最后一个batch的均值和标准差。

BN的好处非常显而易见，一个是可以减少covariate shift。也就是说，以前为了避免每个layer的方差太大，我们会减小步长，但是用了BN以后就可以用大的步长加速训练。此外，对于sigmoid或者tanh这样的激活函数来说，可以有效减少深层网络的梯度爆炸或者消失的问题。另外BN的一个副产物是可以减少过拟合。
