#!/bin/bash
sudo apt install -y zsh

# install the plugins
git clone https://github.com/robbyrussell/oh-my-zsh.git ~/.oh-my-zsh
cp ~/.oh-my-zsh/templates/zshrc.zsh-template ~/.zshrc
# this didnt work
#wget https://github.com/robbyrussell/oh-my-zsh/raw/master/tools/install.sh -O - | zsh
git clone https://github.com/supercrabtree/k ~/.oh-my-zsh/custom/plugins/k
git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ~/.oh-my-zsh/custom/plugins/zsh-syntax-highlighting
git clone https://github.com/zsh-users/zsh-autosuggestions ~/.oh-my-zsh/custom/plugins/zsh-autosuggestions

#set up powerline fonts and the 9k theme
git clone https://github.com/bhilburn/powerlevel9k.git ~/.oh-my-zsh/custom/themes/powerlevel9k
wget https://github.com/powerline/powerline/raw/develop/font/PowerlineSymbols.otf
wget https://github.com/powerline/powerline/raw/develop/font/10-powerline-symbols.conf

sudo apt-get install -y powerline fonts-powerline
#manually install fonts
#mkdir -p ~/.local/share/fonts
#mv PowerlineSymbols.otf ~/.local/share/fonts/
#fc-cache -vf ~/.local/share/fonts/
#mkdir -p ~/.config/fontconfig/conf.d
#mv 10-powerline-symbols.conf ~/.config/fontconfig/conf.d/

#update .zshrc
sed -i '/ZSH_THEME/c\ZSH_THEME="powerlevel9k/powerlevel9k"' ~/.zshrc
echo "plugins+=(git zsh-autosuggestions k  zsh-syntax-highlighting)" >> ~/.zshrc

#set as default shell
MYUSER=`whoami`
sudo -s
chsh -s /bin/zsh root
chsh -s /bin/zsh
chsh -s /bin/zsh $MYUSER
