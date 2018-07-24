---
title: 台大李宏毅机器学习作业——分类算法
categories: 统计学习
mathjax: true
date: 2017-12-26
---

好久没更，李宏毅第二次课程实操

<!-- more -->

第二次作业主要做的是分类算法。分类算法分为Generative和Discriminate两种。Generative常见的比如朴素贝叶斯，Discriminator就多了，逻辑回归就是。这里实现两个不同的分类算法。

首先是probabilistic generative model。概率生成模型其实就是根据数据分布的情况，拟合一种分布，计算条件概率，判断样本的类别。基本公式就是$P(C_i | x) = \frac{P(x | C_i) P(C_i)}{\sum(P(x | C_i) P(C_i))}$。那需要用到概率分布的地方就是$P(x | C_i)$这里，我们需要假设一个分布来拟合这个概率。一般而言，我们用正态分布比较多。之所以用正态分布多，可以回顾一下大数定律和中心极限定理。当数据量到了一定程度，就会呈现出正态分布。

大概的原理可以看这篇[博客](https://samaelchen.github.io/machine_learning_step4/)，不是一个很复杂的模型。这里用到的数据就是[作业2](https://ntumlta.github.io/2017fall-ml-hw2/)的示例数据，是一个二分类的数据。因为模型的公式已经很明显了，这里就直接将公式转化为代码：

```python
def gen_model(X_test, X_train, threshold=0.5):
    """Generative model"""
    D = X_train.shape[1] / 2
    a = X_train.loc[X_train['y'] == 0, :].drop('y', axis=1)
    mu1 = a.mean()
    tmp1 = pd.DataFrame(X_test - mu1)
    b = X_train.loc[X_train['y'] != 0, :].drop('y', axis=1)
    mu2 = b.mean()
    tmp2 = pd.DataFrame(X_test - mu2)
    # 这里的协方差矩阵是两个类别的协方差矩阵的加权平均。通常来说用加权平均的效果比较好。
    sigma = a.shape[0] / len(X) * np.cov(a.T, bias=True) + b.shape[0] / len(X_train) * np.cov(b.T, bias=True)
    sigma_det = 1 / np.sqrt(np.linalg.det(sigma))
    # 这里用伪逆，实际上不用伪逆问题也不大
    sigma_pinv = np.linalg.pinv(sigma)
    # 这里要注意，我直接算的是矩阵，而不是一行一行的算。那么直接计算矩阵的时候其实就只要拿对角线元素就可以了。为什么直接取对角线元素，可以回顾一下MIT的线性代数课程。
    array1 = np.diag(np.dot(tmp1, sigma_pinv).dot(tmp1.T))
    array2 = np.diag(np.dot(tmp2, sigma_pinv).dot(tmp2.T))
    gauss1 = np.power(2 * np.pi, -D) * sigma_det * np.exp(-0.5 * array1)
    gauss2 = np.power(2 * np.pi, -D) * sigma_det * np.exp(-0.5 * array2)
    prob1 = gauss1 * a.shape[0] / len(X)
    prob2 = gauss2 * b.shape[0] / len(X)
    prob = prob1 / (prob1 + prob2)
    prob[np.isnan(prob)] = 0
    # 这里用了一个阈值，其实也可以按照每个类别的先验概率来分类。个人以为比较合理，但是实际上未必效果最好。五五开也是一种常见的方法。
    prob[prob >= threshold] = 0
    prob[prob != 0] = 1
    return prob
```

那实际上这样概率生成模型就结束了。从某种程度而言，概率生成模型是不存在训练测试的。

这里顺带提一下用的几个包：

```python
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import time
```

其实sklearn的几个metric可以自己写，但是这里偷懒了。原则上还是自己实现核心算法。

相对而言，逻辑回归就比较简单了，大概过程跟线性回归是一样的，用梯度下降去找$w$。原理看这篇[博客](https://samaelchen.github.io/machine_learning_step5/)。所以实现也很简单：

```python
def sigma(x):
    return 1 / (1 + np.exp(-x))


def minibatch_sgd(X, y_true, w, eta=0.1, epoch=10, batch_size=100):
    rounds = 0
    while rounds < epoch:
        start = time.time()
        start_index = 0
        sum_error = 0
        while start_index < (X.shape[0] - 1 - batch_size):
            end_index = start_index + batch_size
            if end_index > X.shape[0]:
                end_index = X.shape[0] - 1
            X_batch = X.iloc[start_index:end_index, :]
            error = sigma(np.dot(X_batch, w)) - y_true[start_index:end_index].values
            sum_error += sum(error)
            w -= eta * np.dot(X_batch.T, error)
            start_index += batch_size
        rounds += 1
        end = time.time()
        accuracy = accuracy_score(sigma(np.dot(X, w)), y_true)
        print('epoch: ' + str(rounds) + '  time: '+ str(round(end - start, 2)) + 's' + '   error: ' + str(sum_error) + '  accuracy: ' + str(accuracy))
    return(w)
```

这里用的是minibatch_sgd，速度会快一点。但是不知道为什么我这边跑的每一个epoch的loss都会乱荡，有点尴尬，后面再仔细debug。
