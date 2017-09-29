---
title: 深度学习工作站配置
categories: 深度学习
date: 2017-01-05
---

深度学习工作站配置简介。

<!-- more -->

这是我在意识到已经落后别人太多的情况下，姗姗来迟的深度学习环境搭建。虽然我觉得深度学习已经被过度消费，但是现状是，如果现在还不追赶，就彻底没机会了。大部分的东西都是轻车熟路的，就不赘述了，说几个碰到的坑。

+ 显卡是GTX 1070往上的机器，安装Ubuntu的时候有可能出现显示器无法显示的状况。不同显示器报错不一，解决方案是拆掉独显，用核显装机并安装显卡驱动，最后插回显卡。

+ 装机过程，首先安装系统，固态硬盘挂载系统，机械硬盘挂/home。

+ 装好系统换软件源，一般我选阿里云。之后就是常见的sudo apt update && sudo apt upgrade。

+ 第一件事情，安装小飞机。GFW，用Ubuntu你懂得。
```
sudo add-apt-repository ppa:hzwhuang/ss-qt5
sudo apt-get update
sudo apt-get install shadowsocks-qt5 tsocks
```

+ 第二件事情，搜狗官网下载输入法安装包，注销重新登录生效。
```
sudo apt install gdebi
sudo gdebi sougou.deb
```
+ 后面很多基本上靠谷歌搜，基本上能搞定全部的东西。

+ 多线程运算是个坑。
    + OpenBLAS是个坑，用Ubuntu的好处是用apt安装libopenblas-base和libopenblas-dev后，R、NumPy可以自动调用。但是NumPy要注意一个神坑，所有的BLAS加速仅对float类型有效。
    + MKL是个大坑，安装完，R、NumPy均不能自动调用，需要从源码编译。 R还有一个简单解决方案是安装微软改造的R，原来的RRO，安装完用/path/to/RRO/R/lib下的所有文件替换掉/usr/lib/R/lib下的文件。NumPy的解决方法可以是使用英特尔的ICC编译源码，也可以使用anaconda。MKL效率比OpenBLAS高。但是有原生工具就坚决不用第三方的我表示，MKL我放弃了。

+ 安装CUDA，cuDNN。现在安装CUDA已经非常容易了，deb一装就行。

+ 安装MXNet/TensorFlow。TF已经可以用pip直接安装了，CPU版本就是tensorflow，GPU版本是tensorflow-gpu。MXNet稍微有点搞，需要修改配置文件，但也比较简单，默认直接编译是CPU版本的。好处是MXNet支持R，单纯倒腾数据，其实R的坑比Python少一点，对小白更友好。
