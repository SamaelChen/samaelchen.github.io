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

然后有一点我一直没理解的点就是，在论文里面上一个等式的左边是source layer，右边的是target layer。直观上从forward的方向上看，数据从上一层到下一层，那么变化就应该是第一层经过变化后变到第二层。

论文里面没有太解释为什么会是这样的操作，看了一些别人的博客，大部分人也说得不清不楚的。个人的感觉吧，为了这么做是为了保证输出的feature map的维度能够保持不变，论文里面有一个示例图：
<img src='https://i.imgur.com/XIBMatZ.png'>

从图上面看的话，target和source都是保持不变的，唯一变换的是source上的sampling grid（感觉这么说也不太对，sampling grid的像素点数量其实也没变，就是位置或者说形状变了）。而这个sampling grid就是将target的网格坐标通过上面的公式做仿射变换得到的。那如果反过来，也就是说我们直接用source做放射变换的话，很可能得到的target是不规整的。所以应该说spatial transformer layer做的事情是学习我们正常理解的仿射变换的逆矩阵。比较神奇的是这个用bp居然可以自己学出来。

那么这里就会有个问题，因为sampling grid的像素点其实是没有变过的，所以这就意味着说仿射变换的结果很可能得到是小数的index。比如说$\begin{bmatrix}1.6 \\ 2.4 \end{bmatrix} = \begin{bmatrix}0 & 0.5 \\ 1 & 0 \end{bmatrix} \begin{bmatrix}2 \\ 2 \end{bmatrix} + \begin{bmatrix}0.6 \\ 0.4 \end{bmatrix}$，那么这个时候要怎么办呢？如果我们按照就近原则的话，那么这个位置又会被定位到原图的$[2, 2]$这个位置，那么梯度就会变成0。所以这样是不行的，那么为了可以进行bp，论文里面采用了双线性插值的方法。也就是说，用离这个位置最近的四个顶点的像素，按照距离的比例作为权重，然后加权平均来填补这个位置的像素。

这个算法大概原理如下：

![](https://i.imgur.com/b7IprgN.png)

我们现在想要求中间绿色点的像素，那么我们先算出$R_1$和$R_2$的像素：
$$
R_1 = \frac{x_2 - x}{x_2 - x_1}Q_{11} + \frac{x - x_1}{x_2 - x_1}Q_{21} \\
R_2 = \frac{x_2 - x}{x_2 - x_1}Q_{12} + \frac{x - x_1}{x_2 - x_1}Q_{22}
$$
然后计算$P$的像素：
$$
\boxed{P = \frac{y_2 - y}{y_2 - y_1}R_1 + \frac{y - y_1}{y_2 - y_1}R_2}
$$

那么在DeepMind的试验里面，在卷基层里面加入了ST层之后，收敛以后target得到的输出大体上都是不变的。就像下图：

![](https://i.imgur.com/x0Za3Tx.gif)

另外就是这个变换矩阵，如果我们强行让这个矩阵长成$\begin{bmatrix}1 & 0 & a \\ 0 & 1 & b \end{bmatrix}$，那么就会变成attention模式，网络自己会去原图上面扫描，这样就会知道模型在训练的时候关注图片的哪个位置。看起来就像下图：

![](https://i.imgur.com/IDeic8W.png)

上面那一排的网络有两个ST layer，大体上可以看出来，红色的框都是在鸟头的位置，绿色的框都是在鸟身的位置。

# Highway network & Grid LSTM

哎，每一个课题都不好理解啊，一个一个来吧

to be continue……
