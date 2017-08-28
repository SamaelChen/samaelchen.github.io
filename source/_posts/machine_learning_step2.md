---
title: 台大李宏毅机器学习 02
category: 统计学习
mathjax: true
date: 2017-08-28
---

Now let's figure out what cause the errors.

<!-- more -->

The error of our model comes from **bias** and **variance**.

If we wanna estimate the mean of a variable $x$, assume the mean of $x$ is $\mu$, the variance is $\sigma^2$.

Estimator of mean, sample N points: $\{x^1, x^2, \dots , x^N \}$, $m = \frac{1}{N} \sum_n x^n \ne \mu$, but $E(m) = E\Big(\frac{1}{N} \sum_n x^n \Big) = \frac{1}{N} \sum_n E(x^n) = \mu$ and $Var(m) = \frac{\sigma^2}{N}$. The more data point, the less bias.

Estimator of variance, sample N points: $\{x^1, x^2, \dots , x^N \}$. $s^2 = \frac{1}{N} \sum_n(x^n - m)^2$, it is a biased estimator, which is $E(s^2) = \frac{N-1}{N} \sigma^2$, and the more data points the less bias.

Such that, the bias of model is the distance between the center point of estimators and the target function; the variance of model is the distance between each estimator and the center point of estimators.

But one data set only fit one model, why we have so many estimators? In fact, if we sample different data, that will fit different estimators.

Now if we sample 5000 times, what is the relationship between errors and model's complexity?

<img src=../../images/blog/ml001.png>

And we can see the more complexity the more variance and the less bias. However, the less complexity the more bias and the less variance.

Now let's discuss how to deal with bias and variance.

If we have large bias, we should redesign our model to make it more complex, or add more features.

If we have large variance, we should collect more sample data or regularization.

<img src=../../images/blog/ml002.png>
<img src=../../images/blog/ml003.png>

But we should note that, the larger regularization, the larger bias. This is a trade-off between bias and variance. That means we should split our data into training set and testing set. A more useful way is N-fold cross validation. It works like
<img src=../../images/blog/ml004.png>

The zen of building model, do not focus on training data set too much, then you may get a good estimator on testing data set.
