---
title: Progressive Growing of GANs
categories: 深度学习
mathjax: true
date: 2019-01-24
keywords: [深度学习, CV, GAN, CNN, 对抗生成网络]
---

虽然我没看完李嘉图，但是也没闲着呀，我还是在写pggan的呀。代号屁股gan计划。我是不会承认我想拿pggan去生成大长腿的。

<!-- more -->

这个GAN是NVIDIA在17年发表的[论文](https://arxiv.org/pdf/1710.10196.pdf)，文章写的比较糙。一开始官方放出了Theano的版本，后来更新了基于TensorFlow的版本。都不是我喜欢的框架。然后就看到北大的一位很快做了一个PyTorch的版本。不过写的太复杂了，后面找到的其他版本基本上也写得跟官方的差不多复杂得一塌糊涂。最后找到一个我能看懂，并且很直观的实现方案：[https://github.com/rosinality/progressive-gan-pytorch](https://github.com/rosinality/progressive-gan-pytorch)。然后我就在这个基础上进行修改，做成我比较舒服的脚本。

接下来把几个核心部分做个笔记。

# 两个trick

## Equalized learning rate

作者这里用了第一个trick，就是让每个weight的更新速度是一样的。用的公式是$\hat{w_i} = w_i/c$。其中$w_i$就是权重，而$c$是每一层用何恺明标准化的一个常数。代码如下：

```python
class EqualLR:
    def __init__(self, name):
        self.name = name
    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()
        return weight * np.sqrt(2 / fan_in)

    @staticmethod
    def apply(module, name):
        fn = EqualLR(name)
        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        module.register_forward_pre_hook(fn)
        return fn
    def __call__(self, module, input):
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)


def equal_lr(module, name='weight'):
    EqualLR.apply(module, name)
    return module


class EqualConv2d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        conv = nn.Conv2d(*args, **kwargs)
        conv.weight.data.normal_()
        conv.bias.data.zero_()
        self.conv = equal_lr(conv)

    def forward(self, input):
        return self.conv(input)
```

这个很明显是原来作者写的啦，我大蟒蛇还没这么工程化的水平。

## Pixelwise normalization

这个是在生成器中进行normalization。公式也很简单，就是$b_{x,y} = a_{x,y} / \sqrt{\frac{1}{N} \sum_{j=0}{N-1}(a_{x,y}^j)^2 + \epsilon}$。其中$\epsilon$是一个常数$10^{-8}$，$N$是有多少feature map，$a_{x,y}和b_{x,y}$是原始feature vector和normalize后的feature vector。代码如下：

```python
class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input):
        return input / torch.sqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)
```

这是两个文章重点提出来的trick。其他其实还有很多trick，不过是偏向设计网络结构的。

#PG-GAN主体

接下来就是最核心的部分，生成器和分类器。生成器和分类器的学习方法就是一步步放大图像的尺寸，从$4\times 4$最后放大到$1024 \times 1024$。生成器和分类器也是放大一次增加一个block。而这个block的设计也是参考了resnet，因为突然放大会导致模型不稳定，用这种方法可以平滑过渡。

然后就是PG-GAN和dcgan不一样的地方，dcgan放大的方式是用conv_transpose而PG-GAN用的是上采样的方法。[Deconvolution and Checkerboard Artifacts](https://distill.pub/2016/deconv-checkerboard/)这篇文章讲了为什么用上采样更好，不过我没来得及细看。

所以我们先定义好一个block：

```python
class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel1, pad1, kernel2, pad2, pixel_norm=True):
        super().__init__()

        self.kernel1 = kernel1
        self.kernel2 = kernel2
        self.stride1 = 1
        self.stride2 = 1
        self.pad1 = pad1
        self.pad2 = pad2

        if pixel_norm:
            self.conv = nn.Sequential(EqualConv2d(in_channel, out_channel, self.kernel1, self.stride1, self.pad1),
                                      PixelNorm(),
                                      nn.LeakyReLU(0.2),
                                      EqualConv2d(out_channel, out_channel, self.kernel2, self.stride2, self.pad2),
                                      PixelNorm(),
                                      nn.LeakyReLU(0.2))
        else:
            self.conv = nn.Sequential(EqualConv2d(in_channel, out_channel, self.kernel1, self.stride1, self.pad1),
                                      nn.LeakyReLU(0.2),
                                      EqualConv2d(out_channel, out_channel, self.kernel2, self.stride2, self.pad2),
                                      nn.LeakyReLU(0.2))
    def forward(self, input):
        out = self.conv(input)
        return out
```

## generator

直接上代码：

```python
class Generator(nn.Module):
    def __init__(self, code_dim=512):
        super().__init__()
        self.code_norm = PixelNorm()
        self.progression = nn.ModuleList([ConvBlock(512, 512, 4, 3, 3, 1),
                                          ConvBlock(512, 512, 3, 1, 3, 1),
                                          ConvBlock(512, 512, 3, 1, 3, 1),
                                          ConvBlock(512, 512, 3, 1, 3, 1),
                                          ConvBlock(512, 256, 3, 1, 3, 1),
                                          ConvBlock(256, 128, 3, 1, 3, 1)])
        self.to_rgb = nn.ModuleList([nn.Conv2d(512, 3, 1),
                                     nn.Conv2d(512, 3, 1),
                                     nn.Conv2d(512, 3, 1),
                                     nn.Conv2d(512, 3, 1),
                                     nn.Conv2d(256, 3, 1),
                                     nn.Conv2d(128, 3, 1),])

    def forward(self, input, expand=0, alpha=-1):
        out = self.code_norm(input)
        for i, (conv, to_rgb) in enumerate(zip(self.progression, self.to_rgb)):
            if i > 0 and expand > 0:
                upsample = F.interpolate(out, scale_factor=2)
                out = conv(upsample)
            else:
                out = conv(out)

            if i == expand:
                out = to_rgb(out)

                if i > 0 and 0 <= alpha < 1:
                    skip_rgb = self.to_rgb[i - 1](upsample)
                    out = (1 - alpha) * skip_rgb + alpha * out
                break

        return out
```

这个generator只定义到了$128\times 128$这个分辨率的，要是想要增大分辨率可以参考文章最后的附录table 2的数据自己一个个加上去就好了，discriminator一样的操作就行。然后就是代码里面的这个skip_rgb，这个操作就是上面讲的平滑操作。

## discriminator

跟generator差不多。

```python
class Distriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.progression = nn.ModuleList([ConvBlock(128, 256, 3, 1, 3, 1, pixel_norm=False),
                                          ConvBlock(256, 512, 3, 1, 3, 1, pixel_norm=False),
                                          ConvBlock(512, 512, 3, 1, 3, 1, pixel_norm=False),
                                          ConvBlock(512, 512, 3, 1, 3, 1, pixel_norm=False),
                                          ConvBlock(512, 512, 3, 1, 3, 1, pixel_norm=False),
                                          ConvBlock(513, 512, 3, 1, 4, 0, pixel_norm=False),])
        self.from_rgb = nn.ModuleList([nn.Conv2d(3, 128, 1),
                                       nn.Conv2d(3, 256, 1),
                                       nn.Conv2d(3, 512, 1),
                                       nn.Conv2d(3, 512, 1),
                                       nn.Conv2d(3, 512, 1),
                                       nn.Conv2d(3, 512, 1),])
        self.n_layer = len(self.progression)
        self.linear = nn.Linear(512, 1)

    def forward(self, input, expand=0, alpha=-1):
        for i in range(expand, -1, -1):
            index = self.n_layer - i - 1
            if i == expand:
                out = self.from_rgb[index](input)
            if i == 0:
                mean_std = input.std(0).mean()
                mean_std = mean_std.expand(input.size(0), 1, 4, 4)
                out = torch.cat([out, mean_std], 1)
            out = self.progression[index](out)

            if i > 0:
                out = F.avg_pool2d(out, 2)
                if i == expand and 0 <= alpha < 1:
                    skip_rgb = F.avg_pool2d(input, 2)
                    skip_rgb = self.from_rgb[index + 1](skip_rgb)
                    out = (1 - alpha) * skip_rgb + alpha * out

        out = out.squeeze(2).squeeze(2)
        out = self.linear(out)
        return out
```

然后mean_std这个地方就是文章里面的另一个trick，叫minibatch stddev，主要是用来增加差异性的，文章的第三部分。

最后只要按照wgan的方法训练就好了。不过还要注意一点的就是，wgan是discriminator训练5次，训练一次generator，而pggan是训一次discriminator，一次generator这样交替来。

```python
experiment_path = 'checkpoint/pggan'
img_list = []
G_losses = []
D_losses = []
D_losses_tmp = []
Grad_penalty = []
i = 0
iters = 0
total_iters = 0
expand = 0
n_critic = 1
step = 0
alpha = 0
CLAMP = 0.01
one = torch.FloatTensor([1]).cuda()
mone = one * -1
print('Training start!')
for epoch in range(num_epochs):
    if epoch != 0 and epoch % 2 == 0:
        alpha = 0
        iters = 0
        expand += 1
        if expand >= 3:
            batch_size = 16
        if expand > 5:
            alpha = 1
            expand = 5
        dataset = modify_data(dataroot, image_size * 2 ** expand)
        dataloader = udata.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
    for i, data in enumerate(dataloader):
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        if step < n_critic:
            netD.zero_grad()
            for p in netD.parameters():
                p.requires_grad = True
#                 p.data.clamp_(-CLAMP, CLAMP)
            output = netD(real_cpu, expand, alpha).view(-1)
            errD_real = (output.mean() - 0.001 * (output ** 2).mean()).view(1)
            errD_real.backward(mone)
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake = netG(noise, expand, alpha)
            output = netD(fake.detach(), expand, alpha).view(-1)
            errD_fake = output.mean().view(1)
            errD_fake.backward(one)
            eps = torch.rand(b_size, 1, 1, 1, device=device)
            x_hat = eps * real_cpu.data + (1 - eps) * fake.data
            x_hat.requires_grad = True
            hat_predict = netD(x_hat, expand, alpha)
            grad_x_hat = autograd.grad(outputs=hat_predict.sum(), inputs=x_hat, create_graph=True)[0]
            grad_penalty = ((grad_x_hat.view(grad_x_hat.size(0), -1).norm(2, dim=1) - 1) ** 2).mean()
            grad_penalty = 10 * grad_penalty
            grad_penalty.backward()
            errD = errD_real - errD_fake
            d_optimizer.step()
            D_losses_tmp.append(errD.item())
            step += 1
        else:
            for p in netD.parameters():
                p.requires_grad = False
            netG.zero_grad()
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake = netG(noise, expand, alpha)
            output = netD(fake, expand, alpha).view(-1)
            errG = -output.mean().view(1)
            errG.backward()
            g_optimizer.step()
            D_losses.append(np.mean(D_losses_tmp))
            G_losses.append(errG.item())
            D_losses_tmp = []
            step = 0
        if (total_iters+1) % 200 == 0:
            print('[%d/%d][%d/%d](%d)\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f\tGrad: %.4f'
                  % (epoch+1, num_epochs, i+1, len(dataloader), total_iters + 1,
                     errD.item(), errG.item(), errD_real.data.mean(), errD_fake.data.mean(), grad_penalty.data))
        # Check how the generator is doing by saving G's output on fixed_noise
        if (total_iters % 5000 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise, expand, alpha).detach().cpu()
            img = vutils.make_grid(fake, padding=2, normalize=True)
            vutils.save_image(img, 'checkpoint/pggan/fake_image/fake_iter_{0}.jpg'.format(total_iters))
            img_list.append(img)

        iters += 1
        total_iters += 1
        if (epoch + 1) % 50 == 0:
            torch.save(netG.state_dict(), '{0}/netG_epoch_{1}.pth'.format(experiment_path, epoch+1))
            torch.save(netD.state_dict(), '{0}/netD_epoch_{1}.pth'.format(experiment_path, epoch+1))
```

然后这里面最要注意的是，wgan-gp里面用到了一个很重要的方法，就是gradient penalty，也就是训练里面的这一部分：

```python
eps = torch.rand(b_size, 1, 1, 1, device=device)
x_hat = eps * real_cpu.data + (1 - eps) * fake.data
x_hat.requires_grad = True
hat_predict = netD(x_hat, expand, alpha)
grad_x_hat = autograd.grad(outputs=hat_predict.sum(), inputs=x_hat, create_graph=True)[0]
grad_penalty = ((grad_x_hat.view(grad_x_hat.size(0), -1).norm(2, dim=1) - 1) ** 2).mean()
grad_penalty = 10 * grad_penalty
grad_penalty.backward()
```

别的也就没什么了，坐等结果就好了。具体在我的[notebook](https://github.com/SamaelChen/hexo-practice-code/blob/master/pytorch/pggan/pggan-101.ipynb)里面。
