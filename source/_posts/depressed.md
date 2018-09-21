---
title: 日常的丧
categories: 今天份的不开心
mathjax: true
date: 2018-09-21
keywords: [丧]
---

日常很丧的各种不开心，小确丧。

<!-- more -->

# 2018-09-21

一直在试char-rnn生成，可能notebook硬盘io频繁了一点，终于把工作站的硬盘搞到写保护了。现在整个硬盘全是坏道。情绪稳定。

顺便，Linux检查硬盘坏道的方法：

```bash
sudo badblocks -s -v /dev/sdXX > badblocks.txt
```

-s可以显示检查进度，不过一般显示进度的话实际检查速度貌似会变慢。

然后可以用recovery模式去修复一下，fsck -a /dev/sdXX。运气好是逻辑坏道的话能修复好，如果跟我一样修复好一会儿会儿就又写保护了，估计十有八九是物理坏道。
