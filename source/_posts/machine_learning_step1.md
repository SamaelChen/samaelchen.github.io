---
title: 台大李宏毅机器学习 01
category: 统计学习
mathjax: true
date: 2017-08-26
---

机器学习就是寻找一个函数来解决某一类问题，回归就是预测一个实数的方法。

<!-- more -->

假设我们要预测神奇宝贝的CP值，我们用 $x$ 来表示一只神奇宝贝。用 $x_{cp}$ 表示CP值，用 $x^i$ 表示 $i$th 神奇宝贝。

如果我们想解决这样一个问题，我们可以参照下面的步骤：

Step1. 假设影响神奇宝贝CP值的真实函数是 $y = b + w \cdot x_{cp}$，而集合 $\{f_1, f_2, \dots \}$ 是我们的模型。理论上这个集合包含无数多个函数。这个函数假设 CP 值只受原来 CP 值的影响。一般而言，我们用 $y = b + \sum w_i \cdot x_i$, $x_i$ 表示所有的特征。

Step2. 我们需要找一个评价标准来评价我们模型的好坏。在这个例子中，我们可以用 MSE(Mean Square error)。 $MSE = \sum (y_i - \hat{y}_i)^2$。习惯上我用 $y$ 表示真实值，$\hat{y}$ 表示预测值。这一点跟视频中不太一样，不过只是个标记而已，并不影响理解。实际上有很多类似 MSE 的方程用来衡量预测值与实际值的差距，我们称之为 Loss function。那么第一步中的MSE可以写作
$$L(f) = \sum (y^n - f(x^n))^2 = \sum (y^n - (w \cdot x^n + b))^2.$$

Step3. 现在我们只需要找到损失函数最小的一个假设 $f$。这个例子中我们要做的就是
$$\arg \min L(w, b) = \sum (y^n - (w \cdot x^n + b))^2.$$
这个函数可以用最小二乘来直接解决，但是通常实践上我们用梯度下降来做。我们记损失函数为$L(w, b)$，而且只有一个参数$w$.

+ 首先，我们随机初始化一个 $w^0$
+ 其次，计算 $\frac{\partial{L}}{\partial{w}}|_{w=w^0}$.
+ 第三，更新 $w$。 $w^1 = w^0 - \eta \frac{\partial{L}}{\partial{w}}|_{w=w^0}$, $\eta$ as learning rate.
+ 重复上面三个步骤直到 $\frac{\partial{L}}{\partial{w}}|_{w=w^n} = 0$.

事实上梯度下降是一个典型的贪心算法，很容易停在saddle point或者局部最优解上。通常损失函数非常复杂的时候会这样（当然，理论上会这样，实际上会比这个更惨）。

如果现在有多个参数呢？我们可以用一个向量的方式来表示
$$
\bigtriangledown L = \begin{bmatrix}
\frac{\partial{L}}{\partial{w}} \\
\frac{\partial{L}}{\partial{b}} \\
\end{bmatrix}_{\text{gradient}}.
$$

在这个例子当中，李老师得到了 $\text{error}_{\text{training}} = 31.9$ and $\text{error}_{\text{test}} = 35$.

我们这里讨论了最简单的一个模型，如果现在提高模型复杂度会怎么样呢？如果我们假设 $y = b + w_1 \cdot x + w_2 \cdot x^2$，我们就能得到 $\text{error}_{\text{training}} = 15.4, \ \text{error}_{\text{test}} = 18.4$，看起来好了不少。如果进一步假设 $y = b + w_1 \cdot x + w_2 \cdot x^2 + w_3 \cdot x^3$， 则 $\text{error}_{\text{training}} = 15.3,\  \text{error}_{\text{test}} = 18.1$。

如果现在假设 $x$ 是四次方的一个方程，则 $\text{error}_{\text{training}} = 14.9, \ \text{error}_{\text{test}} = 28.8$. 如果假设$y = b + w_1 \cdot x + w_2 \cdot x^2 + w_3 \cdot x^3 + w_4 \cdot x^4 + w_5 \cdot x^5$，则$\text{error}_{\text{training}} = 12.8，\ \text{error}_{\text{test}} = 232.1$.

当模型复杂度提高的时候，模型在训练集的错误会下降，测试集的也会下降，但是当模型复杂度到了一定程度，随着训练集错误下降，测试集错误上升，我们称之为 **overfitting**。

事实上，解决overfitting的方法可以是使用domain knowledge来设计一个更合理的方程，或者减少模型的复杂度。另一个操作性更强的方法就是增加数据和feature。

比如我们收集了四个神奇宝贝，增加了数据量，那么我们可以把模型设计为：
$$
\begin{align}
y &= b_1 \cdot \delta(x_s=\text{Pidgey}) \\
&+ w_1 \cdot \delta(x_s = \text{Pidgey})x_{cp} \\
&+ b_2 \cdot \delta(x_s = \text{Weedle}) \\
&+ w_2 \cdot \delta(x_s = \text{Weedle})x_{cp} \\
&+ b_3 \cdot \delta(x_s = \text{Caterpie}) \\
&+ w_3 \cdot \delta(x_s = \text{Caterpie})x_{cp} \\
&+ b_4 \cdot \delta(x_s = \text{Eevee}) \\
&+ w_4 \cdot \delta(x_s = \text{Eevee})x_{cp}
\end{align}
$$
其中
$$
\delta(x_s = Pidgey) = \begin{cases}
=1 \quad \text{if } x_s = \text{Pidgey} \\
=0 \quad \text{otherwise}
\end{cases}.
$$

这样我们得到的就是$\text{error}_{\text{training}} = 3.8, \; \text{error}_{\text{test}} = 14.3$.

如果我们继续增加特征或者模型复杂度，有可能再次导致overfitting，所以这里我们对损失函数进行改造：
$$
L = \sum_n \Big(y^n - \big( b + \sum w_i x_i \big) \Big)^2 + \lambda \sum (w_i)^2.
$$
我们增加了一个惩罚项。当模型过于复杂的时候，这个惩罚项会变得很大。因此，越小的系数效果越好。越小的系数意味着越光滑的模型。一般来说我们认为比较光滑的模型更有可能正确。我们将这个方法叫做regularization。这里我们可以看到，我们没有必要对$b$做regularaizaion，因为这个参数仅仅影响模型上下移动，并不影响模型的复杂度。
