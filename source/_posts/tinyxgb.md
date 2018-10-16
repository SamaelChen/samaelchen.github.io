---
title: tiny XGBoost
categories: 统计学习
mathjax: true
date: 2018-10-08
keywords: [机器学习, xgboost, gbdt]
---

XGBoost是GBDT的一个超级加强版，用了很久，一直没细看原理。围观了一波人家的实现，自己也来弄一遍。以后面试上来一句，要不要我现场写个XGBoost，听上去是不是很霸气。

半抄半写，debug起来好痛苦(；′⌒`)

<!--more-->

# 回顾一下ensemble算法

之前也写了一篇笔记，[台大李宏毅机器学习——集成算法](https://samaelchen.github.io/machine_learning_step17/)，最近把一些错误做了修改。

## bagging

bagging的经典算法就是随机森林了，bagging的思路其实非常非常的简单，就是同时随机抽样本和feature，然后建立n个分类器，接着投票就好了。

bagging的做法本质上不会解决bias太大的问题，所以该过拟合还会过拟合。但是bagging会解决variance太大的问题。这个其实是非常直观的一件事情。

然后我突然开了一个脑洞，RF选择feature这种事情吧，如果用一个weight来表示，把树换成逻辑回归，最后voting的事情也用一个weight来表示，感觉，似乎就是单层神经网络的既视感。anyway，无脑揣测，有待探究。

## boosting

boosting跟bagging的套路就不一样，bagging是同时并行很多分类器，但是boosting是串行多个分类器。bagging的分类器之间没有依赖关系，boosting的分类器是有依赖关系的。

boosting算法比较知名的就是GBDT和Adaboost两个。不过其实Adaboost就是一个特殊的GBDT。

GB的一般流程是这样的：

> 初始化一个函数$g_0(x) = 0$ \
> 然后按照迭代次数从$t=1$到$T$循环，我们的目标是找到一个函数$f_t(x)$和权重$\alpha_t$使得我们的函数$g_{t-1}(x)$的效果更好，也就是说：\
> $$g_{t-1}(x) = \sum_{i=1}^{t-1} \alpha_i f_i(x)$$
> 换个角度来看就是$g_t(x) = g_{t-1}(x) + \alpha_t f_t(x)$ \
> 而最后我们优化的损失函数$L(g) = \sum_n l(y_n, g(x_n))$

那Adaboost就是GB的损失函数$l$用exponential表示，也就是$\exp(-y_n g(x_n))$。很美妙的一家子。

那实际上我们看$g_t(x) = g_{t-1}(x) + \alpha_t f_t(x)$这里，非常像梯度下降，那么如果要做梯度下降的话，其实我们就是对$g(x)$做偏导，所以我们得到的是$g_t(x) = g_{t-1}(x) - \eta \frac{\partial L(g)}{\partial g(x)} \bigg|_{g(x)=g_{t-1}(x)}$。那其实只要我们想办法让尾巴后面的那一部分是同一个方向的，我们不就达到了梯度下降的目的了吗？！步子的大小是可以用$\eta$调的，同样这边也可以调整$\alpha$。总之，保证他们的方向一致，就可以做梯度下降了。

### Adaboost

现在将Adaboost的损失函数放进来推算一下：
$$
\begin{align}
L(g) &= \sum_n \exp(-y_n g_t(x_n)) \\
&= \sum_n \exp(-y_n (g_{t-1}(x_n) + \alpha_t f_t(x))) \\
&= \sum_n \exp(-y_n g_{t-1}(x_n)) \exp(-y_n \alpha_t f_t(x_n)) \\
&= \sum_{f_t(x) \ne y} \exp(-y_n g_{t-1}(x_n)) \exp(\alpha_t) + \sum_{f_t(x) = y} \exp(-y_n g_{t-1}(x_n)) \exp(-\alpha_t)
\end{align}
$$

这样一来，我们需要同时寻找一个$\alpha$和$f(x)$使得我们的损失函数最小。找一个最优的$f(x)$这个事情只要让决策树去学习就可以了，那么$\alpha$怎么办呢？如果我们对$\alpha$求偏导，让偏导数为0，那么理论上，我们在$\alpha$这个方向上就是最优的。

具体求偏导这个之前的博客写了，这里就不多搞了。简单说就是刚好会等于$\ln \sqrt{(1-\varepsilon_t) / \varepsilon_t}$，而这个值刚好是每一轮调整样本权重时候的系数取对数。

### GBDT

至于GBDT，实际上跟Adaboost没啥区别，也是一样的搞法，无非是损失函数不一样，优化策略不太一样而已。

GBDT的玩法是用每一棵树去学习上一棵树的残差，通俗的说就是下一棵树学习如何矫正上一棵树的错误。

但是残差这东西也是很妙的一个事情，如果我们损失函数是RMSE，其实我们的残差就是一阶导数。但是呢，如果损失函数是别的函数的时候，其实“残差”这个东西就很难去计算，比如分类的时候，残差究竟是个什么意思？！所以Freidman的梯度提升算法，也就是GBDT前面的GB就是用损失函数对$f(x)$的一阶导数来替代残差的。

总体来说，我们想优化的损失函数是：
$$
\Theta_t = \arg \min_{\Theta_t} \sum_{t=1}^{T} l(y_n, g_{t-1}(x_n) + \alpha_t f_t(x))
$$

如果我们做回归问题，用RMSE为loss function，那么$l(y_n, g_t(x_n))=(y_n - g_t(x_n))^2$，而$y_n - g_t(x_n)$就是传说中的残差。所以用一阶导数来替代残差真的是神来之笔。这样每次的树直接去学习梯度，而在回归的时候残差跟一阶导数还是一样的，美滋滋。

然后突然有个很不成熟的想法，GBDT是不是很怕one hot的feature呢？！举个例子，如果所有的feature都是one hot的，同时我们每个DT都只有一层，是不是理论上来说，最后有多少feature就有多少树。

# XGBoost

嗯，就是一个增强版GBDT。

强在哪呢，boosting是一个加法训练，原来我们用的是一阶导数，而陈天奇则是用了二阶导并加了个正则项。

但是陈天奇做二阶导的目的怎么说呢，我不是非常理解。是inspired by GBDT的RMSE？！这个手算一下，我们的损失函数用RMSE的时候会有一个$(\alpha_t f_t(x))^2$，而这个东西类比泰勒展开的时候就是那个$\Delta x$，所以怎么说呢，可能大牛是因为看到这个时候灵感乍现，然后尝试用了二阶导。

然后在XGBoost里面的损失函数就可以用二阶泰勒展开来表示：
$$
\Theta = \sum_{i=1}^n [g_i f_t(x_i) + \frac{1}{2} h_i f^2_t(x_i)] + \Omega(f_t)
$$
其中的$g_i$和$h_i$分别是一阶导数和二阶导数。这样定义的好处是，以后不论用什么损失函数，总体的优化方向仅仅跟一阶导数和二阶导数有关。那么在这个大框架底下，任意可以二阶导的损失函数都可以放进来了。那么我看了XGB的[论文](https://arxiv.org/pdf/1603.02754.pdf)关于为什么取二阶导的描述：
<img src='https://i.loli.net/2018/10/11/5bbec3573482d.png'>
直观感受上来说吧，就有点像拟牛顿法是梯度下降的一个功能性提升的样子。

另外就是正则项，这里$\Omega(f) = \gamma T + \frac{1}{2} \lambda \sum_{j=1}^T w_j^2$。这里的$T$是我们有多少叶节点，$w$是叶节点的权重。这两个都在XGB的参数里面可以调整的，我们都知道正则项会改变模型的保守程度，或者说就是variance。

其他XGB的优化很多是工程上的优化，这个不是CS科班出身就看不懂了。如何并行化什么的，一脸懵逼。具体可以看陈天奇的论文，数学的部分写的非常美妙。
