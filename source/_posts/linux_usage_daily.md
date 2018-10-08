---
title: Linux 日常使用软件的日经吐槽贴
categories: 搞着玩
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

# 2018.08.15 日经

写hexo用next主题的时候有个问题，就是有时候网页上的icon会不显示，比如home上面那个小房子。其实就是push以后，next文件夹底下的source/lib里少了font-awesome和ua-parser-js两个文件夹。到github的next源码那里复制一份拷下来就好了。这俩文件夹被github屏蔽了难道？导致每次都push不上去？！

# 2018.08.22 日经

atom使用的时候避免回车变成候选词，修改一下keymap：

```
# Disable Enter key for confirming an autocomplete suggestion
'atom-text-editor:not(mini).autocomplete-active':
  'enter': 'editor:newline'
```

另外就是github的二次验证问题。

首先让git记住你的秘钥

```bash
git config --global credential.helper store
git config --global user.email 'XXX@xxx.com'
git config --global user.name 'XXX'
```

接着去github上面拿一个personal token

然后push的时候密码用这个token就可以了。

# 2018.10.08 日经

突然发现不蒜子不能统计数据了，翻了一下发现是作者的七牛云过期了，只要修改一下next/layout/_third_party_analytics/busuanzi-counter.swig里面的js路径就好了。新路径看作者的网站：[https://busuanzi.ibruce.info/]('https://busuanzi.ibruce.info/')
