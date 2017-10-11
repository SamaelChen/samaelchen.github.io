---
title: 台大李宏毅机器学习作业 01
category: 统计学习
mathjax: true
date: 2017-10-11
---

机器学习课程的课程作业。嗯，突然发现一直上理论没有实践，机器学习这样一门实践科学怎么能不实践。

<!-- more -->

课程第一次作业在[这里](
https://docs.google.com/presentation/d/1L1LwpKm5DxhHndiyyiZ3wJA2mKOJTQ2heKo45Me5yVg/edit#slide=id.g1ebd1c9f8d_0_0)

课程用的是kaggle玩的一个数据，预测PM2.5，不过因为不是班上的学生，所以我没法提交，就不用这个数据了。可以用kaggle上面的练手数据来搞。这用house-price这个数据来练手。这样我们可以提交结果，看看自己写的模型跟现有框架的差距。

课程的要求是自己用梯度下降实现一个线性回归，不能用现成的框架，比如Python必备的sklearn，当然同理也不能用MXNet或者TF这样的重武器了。

用梯度下降来实现的话，其实有一个很简单的，重点就是先实现损失函数和梯度下降。秉持写代码就是先写糙活，再做优化的原则，先开始写一个最直觉的函数。
