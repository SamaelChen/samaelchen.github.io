---
title: Linux 日常使用软件的日经吐槽贴
category: 搞着玩
mathjax: true
date: 2018-06-22
---

常年把Linux作为desktop的日经吐槽贴，买MacBook的小钱钱攒够就关了。每个日经贴都是买MacBook的一个理由啊。别问为什么不用Windows，用Python写machine learning的应该知道Windows下面有多折腾。

等买MacBook的小钱钱攒够了就关掉此贴（大概此生无望了/(ㄒoㄒ)/~~）。

<!--more-->

说一下用过的发行版，Ubuntu启蒙，目前用的是Ubuntu，另外就是国产的Deepin。不过说起来，Ubuntu放弃了Unity，Deepin创始人离开，感觉Linux没什么可以留恋的了。哪天Unity彻底死掉，就弃坑了。

在国内能够愉快用Linux不得不感谢Deepin的付出。首先是国内绕不开的QQ和微信，Deepin wine都有很好的支持。此外，Deepin共同开发了搜狗输入法（其实是Deepin最早开始的，后来优麒麟不要脸抢走了的样子），网易云音乐和有道词典。

# 2018.06.22 日经

这次日经贴就是吐槽词典的，Deepin毕竟人手有限，有道词典有个可怕的bug，就是内存泄露，这个程序已经3年没更新了，所以这个bug就一直在。

那为了愉快使用词典，我寻求了古老但坚挺的GoldenDict。在Ubuntu下面，老将还是非常稳定的。但是在最新的Deepin 15.6下面，GoldenDict会莫名其妙一直占满一个线程（WTF）。

于是只能转寻命令行查词的工具了。一开始用的是dictd，非常好用，但是不支持扩展星际译王的词典，所以放弃了。

然后用的是sdcv，星际译王的终端版本，支持扩展词典，效果棒棒哒。目前在用，效果如图：

<img src='https://i.imgur.com/EQg93QK.jpg'>

安装方法很简单：
```bash
sudo apt install sdcv
```
然后将自己下载好的辞书放到～/.stardict/dic路径下就可以了。嗯，不折腾买MacBook理由+1。

# 2018.07.09 日经

这个感觉也不是Linux的问题，主要是markdown底下碰到了要打中文间隔号的时候，发现诶，好像不是很好用啊。如果跟我一样用的是搜狗输入法，那么可以输入yd，然后就有这个符号的候选了。感谢搜狗。
