---
title: 台大李宏毅机器学习作业——CNN visualization
categories: 统计学习
date: 2018-02-01
---

基于MXNet的入门级CNN visualization。嗯TF无脑黑，MXNet & PyTorch一生推。

<!-- more -->

具体的可以去看这里的[ipython notebook](https://github.com/SamaelChen/hexo-practice-code/blob/master/mxnet/mxnet-101-4.ipynb)，可以直接跑的。

```python
import numpy as np
import os
import mxnet as mx
from mxnet import gluon
from mxnet import image
from mxnet import nd
from mxnet import init
from mxnet import autograd
from mxnet.gluon.data import vision
from mxnet.gluon import nn
from mxnet.gluon.model_zoo import vision as models
from PIL import Image
from jupyterthemes import jtplot
jtplot.style(theme='onedork', grid=False)
```


```python
%matplotlib inline
import matplotlib.pyplot as plt
```

修改模型下载源，对国内下载速度友好一点


```python
os.environ['MXNET_GLUON_REPO']='https://apache-mxnet.s3.cn-north-1.amazonaws.com.cn/'
```

加载三个预训练好的模型


```python
vgg19 = models.vgg19(pretrained=True)
vgg16 = models.vgg16(pretrained=True)
resnet152 = models.resnet152_v1(pretrained=True)
```

这里读取一张妹子图片


```python
data = nd.array(np.asarray(Image.open('000000.jpg')))
```


```python
plt.imshow(np.asarray(Image.open('000000.jpg')))
```

<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_8_1.png>

这里要注意，预训练好的模型输入的图片大小是$224 \times 224$的，因此这里对图片重新进行缩放，另外因为MXNet的输入格式有需求，所以我们这里也做了reshape的动作。


```python
data = mx.image.imresize(data, 224, 224)
data = nd.transpose(data, (2, 0, 1))
# data = data.astype(np.float32)/127.5-1
data = data.astype(np.float32)/255
data = data.reshape((1,)+data.shape)
print(data.shape)
```

    (1, 3, 224, 224)


接下来要画的是Saliency Map。可以参考这篇[论文](https://arxiv.org/pdf/1312.6034.pdf)。实际上就是看哪个位置的梯度最大。


```python
data.attach_grad()
with autograd.record():
    out = vgg19(data)
out.backward()
```


```python
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow((data[0].asnumpy().transpose(1, 2, 0)*255).astype(np.uint8))
# plt.imshow(((data[0].asnumpy().transpose(1, 2, 0)+1)*127.5).astype(np.uint8))
plt.subplot(1, 3, 2)
plt.imshow(np.abs(data.grad.asnumpy()[0]).max(axis=0), cmap='gray')
plt.subplot(1, 3, 3)
plt.imshow(np.abs(data.grad.asnumpy()[0]).max(axis=0), cmap=plt.cm.jet)
```


<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_13_1.png>



```python
data.attach_grad()
with autograd.record():
    out = resnet152(data)
out.backward()
```


```python
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow((data[0].asnumpy().transpose(1, 2, 0)*255).astype(np.uint8))
# plt.imshow(((data[0].asnumpy().transpose(1, 2, 0)+1)*127.5).astype(np.uint8))
plt.subplot(1, 3, 2)
plt.imshow(np.abs(data.grad.asnumpy()[0]).max(axis=0), cmap='gray')
plt.subplot(1, 3, 3)
plt.imshow(np.abs(data.grad.asnumpy()[0]).max(axis=0), cmap=plt.cm.jet)
```


<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_15_1.png>


这里有个很有意思的现象，VGG-19偏向找人的头部区域，而ResNet则是找到了腿。另外可以多试验几张图，看看效果。一般试下来VGG偏向把轮廓弄出来，ResNet就会找到各种奇奇怪怪的地方去。但是ResNet效果很好，暂时不能理解为什么。

接下来我们把filter画出来，先看一下VGG的结构。


```python
print(vgg19)
```

    VGG(
      (features): HybridSequential(
        (0): Conv2D(3 -> 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): Activation(relu)
        (2): Conv2D(64 -> 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (3): Activation(relu)
        (4): MaxPool2D(size=(2, 2), stride=(2, 2), padding=(0, 0), ceil_mode=False)
        (5): Conv2D(64 -> 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (6): Activation(relu)
        (7): Conv2D(128 -> 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (8): Activation(relu)
        (9): MaxPool2D(size=(2, 2), stride=(2, 2), padding=(0, 0), ceil_mode=False)
        (10): Conv2D(128 -> 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (11): Activation(relu)
        (12): Conv2D(256 -> 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (13): Activation(relu)
        (14): Conv2D(256 -> 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (15): Activation(relu)
        (16): Conv2D(256 -> 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (17): Activation(relu)
        (18): MaxPool2D(size=(2, 2), stride=(2, 2), padding=(0, 0), ceil_mode=False)
        (19): Conv2D(256 -> 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (20): Activation(relu)
        (21): Conv2D(512 -> 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (22): Activation(relu)
        (23): Conv2D(512 -> 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (24): Activation(relu)
        (25): Conv2D(512 -> 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (26): Activation(relu)
        (27): MaxPool2D(size=(2, 2), stride=(2, 2), padding=(0, 0), ceil_mode=False)
        (28): Conv2D(512 -> 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (29): Activation(relu)
        (30): Conv2D(512 -> 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (31): Activation(relu)
        (32): Conv2D(512 -> 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (33): Activation(relu)
        (34): Conv2D(512 -> 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (35): Activation(relu)
        (36): MaxPool2D(size=(2, 2), stride=(2, 2), padding=(0, 0), ceil_mode=False)
        (37): Dense(25088 -> 4096, Activation(relu))
        (38): Dropout(p = 0.5)
        (39): Dense(4096 -> 4096, Activation(relu))
        (40): Dropout(p = 0.5)
      )
      (https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output): Dense(4096 -> 1000, linear)
    )


把每个卷积层的权重打出来。


```python
for i in vgg19.features:
    if isinstance(i, nn.Conv2D):
        j = i.weight.data()
        print(i.weight.data()[0])
```

将最后一层卷积层的第一个filter画出来，然而，完全看不出到底这个filter能起到什么效果。


```python
plt.imshow(np.abs(j[0][0].asnumpy()))
```


<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_23_1.png>


取第一层卷积层出来


```python
for i in vgg19.features:
    if isinstance(i, nn.Conv2D):
        j = i
        print(i.weight.data()[0])
        break
```


    [[[-0.05347426 -0.04925704 -0.06794177]
      [ 0.01531445  0.04506842  0.0021444 ]
      [ 0.03622622  0.01999945  0.01986402]]

     [[ 0.01701478  0.05540261 -0.0062293 ]
      [ 0.14164735  0.22705214  0.13758276]
      [ 0.12000094  0.2002953   0.09211431]]

     [[-0.04488515  0.01267995 -0.01449722]
      [ 0.05974238  0.13954678  0.05410246]
      [-0.00096141  0.058304   -0.02966315]]]
    <NDArray 3x3x3 @cpu(0)>


看一张图片进入第一个卷积层后会得到什么样的结果


```python
for num in range(64):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(np.abs(i(data)[0].asnumpy())[num])
```

    /home/samael/anaconda3/lib/python3.6/site-packages/matplotlib/pyplot.py:528: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      max_open_warning, RuntimeWarning)



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_28_1.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_28_2.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_28_3.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_28_4.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_28_5.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_28_6.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_28_7.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_28_8.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_28_9.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_28_10.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_28_11.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_28_12.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_28_13.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_28_14.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_28_15.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_28_16.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_28_17.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_28_18.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_28_19.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_28_20.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_28_21.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_28_22.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_28_23.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_28_24.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_28_25.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_28_26.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_28_27.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_28_28.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_28_29.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_28_30.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_28_31.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_28_32.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_28_33.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_28_34.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_28_35.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_28_36.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_28_37.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_28_38.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_28_39.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_28_40.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_28_41.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_28_42.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_28_43.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_28_44.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_28_45.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_28_46.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_28_47.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_28_48.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_28_49.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_28_50.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_28_51.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_28_52.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_28_53.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_28_54.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_28_55.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_28_56.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_28_57.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_28_58.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_28_59.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_28_60.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_28_61.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_28_62.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_28_63.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_28_64.png>


然后，嗯，很神奇，第一个卷积的64个通道的效果都在上面。中间有一些看上去还有点像浮雕的效果。某一张嘴唇位置及其显眼。


```python
sample = np.random.uniform(150, 180, (224, 224, 3))
```


```python
plt.imshow(sample)
```




    <matplotlib.image.AxesImage at 0x7f4ec0496e10>




<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_31_1.png>


这里生成一张充满噪声的点，再来看看每个filter在做什么。


```python
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
for channel in range(3):
    sample[:, :, channel] /= 255
    sample[:, :, channel] -= mean[channel]
    sample[:, :, channel] /= std[channel]

sample = sample.reshape((1,)+sample.shape)
sample = sample.transpose(0, 3, 1, 2)
sample = nd.array(sample)
```


```python
for num in range(64):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(np.abs(i(sample)[0].asnumpy())[num])
```

    /home/samael/anaconda3/lib/python3.6/site-packages/matplotlib/pyplot.py:528: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      max_open_warning, RuntimeWarning)



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_34_1.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_34_2.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_34_3.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_34_4.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_34_5.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_34_6.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_34_7.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_34_8.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_34_9.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_34_10.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_34_11.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_34_12.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_34_13.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_34_14.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_34_15.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_34_16.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_34_17.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_34_18.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_34_19.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_34_20.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_34_21.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_34_22.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_34_23.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_34_24.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_34_25.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_34_26.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_34_27.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_34_28.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_34_29.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_34_30.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_34_31.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_34_32.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_34_33.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_34_34.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_34_35.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_34_36.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_34_37.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_34_38.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_34_39.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_34_40.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_34_41.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_34_42.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_34_43.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_34_44.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_34_45.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_34_46.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_34_47.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_34_48.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_34_49.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_34_50.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_34_51.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_34_52.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_34_53.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_34_54.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_34_55.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_34_56.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_34_57.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_34_58.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_34_59.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_34_60.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_34_61.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_34_62.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_34_63.png>



<img src=https://raw.githubusercontent.com/SamaelChen/samaelchen.github.io/hexo/images/blog/output_34_64.png>


入门级别的CNN visualization基本上就这些了。网上没找到MXNet做这个的教程，只能自己摸索了。还好gluon跟pytorch接口很像，可以照着MXNet的源码，再借鉴pytorch的教程慢慢摸索。
