---
title: tiny XGBoost
categories: 统计学习
mathjax: true
date: 2018-10-08
keywords: [机器学习, xgboost, gbdt]
---

XGBoost是GBDT的一个超级加强版，用了很久，一直没细看原理。围观了一波人家的实现，自己也来弄一遍。以后面试上来一句，要不要我现场写个XGBoost，听上去是不是很霸气。

别点了，就是留个坑o(>﹏<)o

<!--more-->

# 回顾一下ensemble算法

之前也写了一篇笔记，[台大李宏毅机器学习——集成算法](https://samaelchen.github.io/machine_learning_step17/)，最近把一些错误做了修改。

## bagging

bagging的经典算法就是随机森林了，bagging的思路其实非常非常的简单，就是同时随机抽样本和feature，然后建立n个分类器，接着投票就好了。

bagging的做法本质上不会解决bias太大的问题，所以该过拟合还会过拟合。但是bagging会解决variance太大的问题。
