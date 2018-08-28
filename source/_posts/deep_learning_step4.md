---
title: 台大李宏毅深度学习——神经网络一些骚操作
categories: 深度学习
mathjax: true
date: 2018-07-23
keywords: [深度学习, spatial transformer, highway network, recursive network]
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

<p align='center'>
<img src='https://i.imgur.com/XIBMatZ.png' width=70%>
</p>

从图上面看的话，target和source都是保持不变的，唯一变换的是source上的sampling grid（感觉这么说也不太对，sampling grid的像素点数量其实也没变，就是位置或者说形状变了）。而这个sampling grid就是将target的网格坐标通过上面的公式做仿射变换得到的。那如果反过来，也就是说我们直接用source做放射变换的话，很可能得到的target是不规整的。所以应该说spatial transformer layer做的事情是学习我们正常理解的仿射变换的逆矩阵。比较神奇的是这个用bp居然可以自己学出来。

那么这里就会有个问题，因为sampling grid的像素点其实是没有变过的，所以这就意味着说仿射变换的结果很可能得到是小数的index。比如说$\begin{bmatrix}1.6 \\ 2.4 \end{bmatrix} = \begin{bmatrix}0 & 0.5 \\ 1 & 0 \end{bmatrix} \begin{bmatrix}2 \\ 2 \end{bmatrix} + \begin{bmatrix}0.6 \\ 0.4 \end{bmatrix}$，那么这个时候要怎么办呢？如果我们按照就近原则的话，那么这个位置又会被定位到原图的$[2, 2]$这个位置，那么梯度就会变成0。所以这样是不行的，那么为了可以进行bp，论文里面采用了双线性插值的方法。也就是说，用离这个位置最近的四个顶点的像素，按照距离的比例作为权重，然后加权平均来填补这个位置的像素。

这个算法大概原理如下：

<p align='center'>
<img src='https://i.imgur.com/b7IprgN.png' width=50%>
</p>

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

<p align='center'>
![](https://i.imgur.com/x0Za3Tx.gif)
</p>

另外就是这个变换矩阵，如果我们强行让这个矩阵长成$\begin{bmatrix}1 & 0 & a \\ 0 & 1 & b \end{bmatrix}$，那么就会变成attention模式，网络自己会去原图上面扫描，这样就会知道模型在训练的时候关注图片的哪个位置。看起来就像下图：

<p align='center'>
<img src='https://i.imgur.com/IDeic8W.png' width=70%>
</p>

上面那一排的网络有两个ST layer，大体上可以看出来，红色的框都是在鸟头的位置，绿色的框都是在鸟身的位置。

# Highway network & Grid LSTM

Highway network最早是[Highway Networks](https://arxiv.org/pdf/1505.00387.pdf)和[Training Very Deep Networks](https://arxiv.org/pdf/1507.06228.pdf)这两篇论文提出的。Highway network实际上受到了LSTM的启发，从结构上来看，深层的前馈网络其实和LSTM非常的像，如下图：

<p align='center'>
<img src='https://i.imgur.com/L4cqtdk.png' width=80%>
</p>

所以二者的差别就在于，在前馈中只有一个input，而LSTM中每一层都要把这一个时刻的x也作为输入。所以很自然的一个想法，在LSTM中有一个forget gate决定要记住以前多久的信息，那么在前馈网络中也可以引入一个gate来决定有哪些之前的信息干脆就不要了，又或者有哪些以前的信息直接在后面拿来用。那最简单LSTM变种是GRU，所以highway network借鉴了GRU的方法，把reset gate拿掉，再把每个阶段的x拿掉。

所以将GRU简化一下再竖起来，我们就可以得到highway network：

<p align='center'>
<img src='https://i.imgur.com/SsDSDuy.png' width=70%>
</p>

那么模仿GRU的计算方法，我们计算$h' = \sigma(Wa^{t-1})$，$z = \sigma(W^z a^{t-1})$，所以$a^t = z \odot a^{t-1} + (1-z) \odot h'$。

而后面微软的[ResNet](https://arxiv.org/pdf/1512.03385.pdf)其实就是一个highway network的特别版本：

<p align='center'>
<img src='https://i.imgur.com/hDTBRrE.png' width=60%>
</p>

当然感觉也可以将ResNet看做是竖起来的LSTM。那ResNet里面的变换可以是很多层的，所以在现在的实现中，很常见的一个情况是将这个东西叫做一个residual block。

所以利用highway network有一个非常明显的好处就是可以避免前馈网络太深的时候会导致梯度消失的问题。另外有一个好处就是通过highway network可以让网络自己去学习到底哪个layer是有用的。

那既然可以将深度的记忆传递下去，那么这样的操作也可以用到LSTM里面，也就是grid LSTM。一般的LSTM是通过forget gate将时间方向上的信息传递下去的，但是并没有将layer之间的信息传递下去。因此grid LSTM就是加一个参数纵向传递，从而将layer的信息传递下去，直观上来说，就是在$y$后面再拼一个vector，然后这个vector的作用跟$c$一样。具体的可以看一下DeepMind的这篇论文，[Grid LSTM](https://arxiv.org/pdf/1507.01526v1.pdf)。粗略来说，结构上像这样：

<p align='center'>
<img src='https://i.imgur.com/BUWr2kn.png' width=60%>
</p>

那有2D的grid LSTM很自然就会有3D的grid LSTM，套路都是差不多的。不过我还没想到的是，3D的grid LSTM要用在什么场景当中，多个output？！

# Recursive Structure

遥想当年刚接触RNN的时候根本分不清recursive network和recurrent network，一个是递归神经网络，一个是循环神经网络，傻傻分不清。但是实际上，recurrent network可以看作是recursive network的特殊结构。Recursive network本身是需要事先定义好结构的，比如：

<p align='center'>
<img src='https://i.imgur.com/SgEEBbw.png' width=60%>
</p>

那常见的recurrent network其实也可以看做是这样的一个树结构的recursive network。Recursive network感觉上好像也没什么特别有意思的东西，比较有趣的就是这边$f$的设计。比如说现在想要让机器学会做句子的情感分析，那么很简单的一个想法就是把每一个词embedding，然后放到网络里面训练，那么我们可以用这样的一个结构：

<p align='center'>
<img src='https://i.imgur.com/YFhqVos.png' width=60%>
</p>

因为在自然语言里面会有一些类似否定之否定的语法，所以我们希望说very出现的时候是加强语气，但是not出现的时候就是否定之前的。如果用数学的语言来表达，这不就是乘以一个系数嘛。所以在这样的情况下，如果我们只是简单的加减操作，那么就没有办法实现这种“乘法”操作。所以这个时候，我们的$f$设计就会有点技巧：

<p align='center'>
<img src='https://i.imgur.com/E4Gt3UC.png' width=60%>
</p>

那么看一下这个设计。如果我们直接采用最传统的做法，就是将$a$和$b$直接concat起来，然后乘以一个矩阵$W$，再经过一个激活函数变换，这样的操作其实只能做到线性的关系，个人感觉，实际上这样的设计会将$a$和$b$的一些交互特性变成隐藏特征保存在$W$当中，但是一旦输入变化了，这些隐藏的特征却不能被传递出来，所以效果不好。

因此下面的一种设计就比较骚气，后面还是传统的做法，但是前面加上了一个vector。这个vector的元素就是来学习这些词之间的interaction，然后将这些interaction变成bias，因为recursive network的function都是不变的，因此这些bias就这样被传递下去了。那么这里有一个需要注意的就是，我们这里有几个词，那我们就需要多少个bias，而且每个bias中间的这个矩阵$W$都是不一样的。

这是三个比较骚气的网络结构变换，感觉看了这么多，好多网络之间都是殊途同归啊，会不会最后有一个非常general的网络结构出现，使得现在的每一种网络都是其一种特殊情况呢？
