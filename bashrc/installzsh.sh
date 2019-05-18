!#/bin/bash
sudo apt install zsh
wget https://github.com/robbyrussell/oh-my-zsh/raw/master/tools/install.sh -O - | zsh
git clone https://github.com/supercrabtree/k $ZSH_CUSTOM/plugins/k
git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting
git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions
#9k shell theme
git clone https://github.com/bhilburn/powerlevel9k.git ~/.oh-my-zsh/custom/themes/powerlevel9k
sed -i '/ZSH_THEME/c\ZSH_THEME="powerlevel9k/powerlevel9k"' ~/.zshrc
mkdir .local/share/fonts
mv PowerlineSymbols.otf ~/.local/share/fonts/
fc-cache -vf ~/.local/share/fonts/
mkdir .config/fontconfig
mkdir .config/fontconfig/conf.d
mv 10-powerline-symbols.conf ~/.config/fontconfig/conf.d/
echo "plugins+=(git zsh-autosuggestions k  zsh-syntax-highlighting)" >> ~/.zshrc
MYUSER=`whoami`
sudo -s
chsh -s /bin/zsh root
chsh -s /bin/zsh
chsh -s /bin/zsh $MYUSER
