---
title: Ubuntu 装机后必搞的一些软件
category: 搞着玩
mathjax: true
date: 2018-06-22
---

记录一下装机后必搞的一些软件，以后装机简单一点。

<!-- more -->

每次装机后都要装好多软件，这里记录一下，以后就简单多了。

添加PPA：

```bash
sudo add-apt-repository ppa:hzwhuang/ss-qt5
sudo add-apt-repository ppa:webupd8team/java
sudo add-apt-repository ppa:jfi/ppa
sudo add-apt-repository ppa:indicator-multiload/stable-daily
sudo add-apt-repository ppa:noobslab/themes
sudo add-apt-repository ppa:noobslab/icons

wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | sudo apt-key add
sudo sh -c 'echo "deb http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google-chrome.list'

curl -sL https://packagecloud.io/AtomEditor/atom/gpgkey | sudo apt-key add -
sudo sh -c 'echo "deb [arch=amd64] https://packagecloud.io/AtomEditor/atom/any/ any main" > /etc/apt/sources.list.d/atom.list'

wget -q -O - http://archive.getdeb.net/getdeb-archive.key | sudo apt-key add -
sudo sh -c 'echo "deb http://archive.getdeb.net/ubuntu xenial-getdeb apps" >> /etc/apt/sources.list.d/getdeb.list'
```

然后更新安装：

```bash
sudo apt update

sudo apt install git terminator guake shadowsocks-qt5 nethogs dpkg oracle-java8-installer atom screenfetch sensord lm-sensors hddtemp psensor indicator-multiload shutter kazam vlc okular ubuntu-tweak flatabulous-theme ultra-flat-icons tsocks vim google-chrome
```

装完小飞机后修改tsocks的配置文件
```bash
sudo vim /etc/tsocks.conf
```
找到 server 改成 server = 127.0.0.1。

然后是百度搜狗输入法，网易云音乐安装包。用dpkg安装就好了。

然后是搜cuda，按照官方的步骤一步步安装。另外就是安装anaconda。这个也不赘述。

接着是一些骚兮兮的美化工作。用Ubuntu-tweak把系统主题改成flatabulous，图标改成ultra-flat-icons。不过可惜啊，flatabulous的作者不再继续支持这个主题了，感觉以后要用macbuntu之类的了（这两天逛gnome-look.org发现另一个挺好看的主题，ant themes，都是黑色的主题，简直本命）。

接着是设置一下psensor，在toolbar显示CPU和GPU的温度。

这些弄完，配置一下atom作为Python的ide：
```bash
apm install linter hydrogen markdown-preview-enhanced atom-beautify language-markdown language-latex atom-language-r project-manager
pip install flake8 flake8-docstrings
apm install linter-flake8
pip install autopep8
```

然后是在preference里面把tab length改成4个spaces。另外为了避免enter自动补全，而只是换行，修改keymap，添加：
```
# Disable Enter key for confirming an autocomplete suggestion
'atom-text-editor:not(mini).autocomplete-active':
  'enter': 'editor:newline'
```

另外atom-beautify的快捷键是ctrl+alt+B，跟fcitx的软键盘快捷键冲突了，吧fcitx开启软键盘的快捷键改掉就好了。

然后是安装zsh，配置一个骚气的终端。

```bash
sudo apt install zsh
chsh -s /bin/zsh
```

接着安装oh-my-zsh

```bash
sh -c "$(curl -fsSL https://raw.github.com/robbyrussell/oh-my-zsh/master/tools/install.sh)"
```

配置自动跳转
```bash
sudo apt-get install autojump
vim .zshrc
#在最后一行加入，注意点后面是一个空格
. /usr/share/autojump/autojump.sh
```

配置语法高亮
```bash
git clone https://github.com/zsh-users/zsh-syntax-highlighting.git
echo "source ${(q-)PWD}/zsh-syntax-highlighting/zsh-syntax-highlighting.zsh" >> ${ZDOTDIR:-$HOME}/.zshrc
```

配置语法历史记录
```bash
git clone git://github.com/zsh-users/zsh-autosuggestions $ZSH_CUSTOM/plugins/zsh-autosuggestions
```
然后修改.zshrc，改成
```
plugins=(
  git
  zsh-autosuggestions
)
```
然后在最后一行加入
```
source $ZSH_CUSTOM/plugins/zsh-autosuggestions/zsh-autosuggestions.zsh
```

最后是配置一个骚气的主题，可以看[https://github.com/robbyrussell/oh-my-zsh/wiki/External-themes](https://github.com/robbyrussell/oh-my-zsh/wiki/External-themes)里面的主题，这里我用的是agnosterzak。

先安装powerline：
```bash
sudo apt install fonts-powerline
```
然后将主题放到~/.oh-my-zsh/themes下面：
```bash
wget http://raw.github.com/zakaziko99/agnosterzak-ohmyzsh-theme/master/agnosterzak.zsh-theme -P ~/.oh-my-zsh/themes
```
然后修改
```
ZSH_THEME="agnosterzak"
```
最后将.bashrc底下一些新加入的export啊，alias啊什么的复制到.zshrc底下就好了。

退出终端重新进入，骚气的zsh就配置好了。

然后是电池电量不显示的问题，安装acpi就可以显示。

如果想卸载的话，执行：
```bash
sudo sh -c "$(curl -fsSL https://raw.github.com/robbyrussell/oh-my-zsh/master/tools/uninstall.sh)"
```

然后将/etc/passwd里面找到自己用户名那一行，把/usr/bin/zsh改成/bin/bash/就可以了。

还有就是配置一个自我发泄的自动对命令行纠错的插件，thefuck，前面配好anaconda以后，只要pip install thefuck就行。然后在.zshrc最后加上eval $(thefuck --alias)。后面如果敲错代码，只要输入fuck，就会自动纠错。

另外，作为Unity的死忠粉，如果装的是Ubuntu 18.04，那么更新完软件以后，要做的事情就是：
```bash
sudo apt install ubuntu-unity-desktop
```
记着将display manager改成lightdm就好了。

目前大概就是这样吧，以后想到有什么更新再说。
