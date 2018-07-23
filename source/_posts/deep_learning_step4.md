---
title: 台大李宏毅深度学习——神经网络一些骚操作
category: 深度学习
mathjax: true
date: 2018-07-23
---

神经网络自2012年躲过世界末日后开始寒武纪大爆发的既视感，从最原始的FC开始出现了各种很风骚的操作。这里介绍三种特殊的结构：spatial transformer layer，highway network & grid LSTM，还有recursive network。

<!-- more -->

# Spatial Transformer

CNN是这一次深度学习大爆发的导火线，但是CNN有非常明显的缺陷。如果一个图像里的元素发生了旋转、位移、缩放等等变换，那么CNN的filter就不认识同样的元素了。也就是说，对于CNN而言，如果现在训练数字识别，喂进去的数据全是规整的，那么倾斜的数字可能就不认识了。

其实从某种意义上来说，这个就是过拟合了，每个filter能做的事情是非常固定的。不过换个角度来看，是不是也能理解为数据过分干净了？

那么为了解决这样的问题，其实有很多解决方案，比如说增加样本量是最简单粗暴的方法，通过image augment就可以得到海量的训练数据。另外一般CNN里面的pooling层也是解决这个问题的，不过受限于pooling filter的size大小，一般来说很难做到全图级别的transform。另外一种做法就是spatial transformer。实际上，spatial transformer layer我感觉上就是嵌入网络的image augment，或者说是有导向性的image augment。这样的做法可以减少无脑augment带来太多的噪音。个人理解不一定对。这种方法是DeepMind提出的，论文就是[Spatial Transformer Networks](https://arxiv.org/pdf/1506.02025.pdf)。

首先看一下如果要对图像进行transform的操作，我们应该怎么做？对于一个图像里的像素而言有两个下标$(x,y)$来表示位置，那么我们就可以将这个看作是一个向量。这样以来，我们只需要通过一个二阶方阵就可以操作图像的缩放和旋转，然后加上一个二维向量就可以控制图片的平移。也就是说
$$\begin{bmatrix}x' \\ y' \end{bmatrix} = \begin{bmatrix}a & b \\ c & d \end{bmatrix} \begin{bmatrix}x \\ y \end{bmatrix} + \begin{bmatrix}e \\ f \end{bmatrix}$$

当然，简洁一点可以写成：
$$
\begin{bmatrix}x' \\ y' \end{bmatrix} = \begin{bmatrix}a & b & c \\ d & e & f \end{bmatrix} \begin{bmatrix}x \\ y \\ 1 \end{bmatrix}
$$
两个公式的元素没有严格对应，不过意思一样。

但是这里需要注意的事情是，比如我们原来输入的图片是$3 \times 3$的，我们输出的还是一个$3 \times 3$的图片，而位置变换后超出下标的部分我们就直接丢弃掉，而原来有值，现在没有值的部分就填0。示意图如下：

<img src='https://i.imgur.com/CSnxAU6.png'>

然后有一点我一直没理解的点就是，在论文里面
$$
\begin{bmatrix}x' \\ y' \end{bmatrix} = \begin{bmatrix}a & b & c \\ d & e & f \end{bmatrix} \begin{bmatrix}x \\ y \\ 1 \end{bmatrix}
$$
这个等式的左边是source layer，右边的是target layer。直观上从forward的方向上看，数据从上一层到下一层，那么变化就应该是第一层经过变化后变到第二层。
