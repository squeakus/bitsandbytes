#!/bin/bash
sudo -E add-apt-repository -y ppa:xorg-edgers/ppa
sudo -E add-apt-repository -y ppa:webupd8team/atom
sudo apt-get update

sudo apt-get -y dist-upgrade
sudo apt-get -y install aptitude git subversion vim compizconfig-settings-manager compiz-plugins compiz-plugins-extra python3

git config --global user.email "jonathanbyrn@gmail.com"
git config --global user.name "squeakus"
git config --global push.default simple
git config --global http.proxy http://proxy-chain.intel.com:911
git config --global credential.helper 'cache --timeout=360000'
git config core.fileMode false


sudo aptitude -y install nodejs npm
sudo ln -s /usr/bin/nodejs /usr/bin/node
npm config set proxy http://proxy-chain.intel.com:911
sudo npm install -g -y http-server
sudo aptitude -y install openssh-server
sudo aptitude -y install fail2ban
sudo aptitude -y install indicator-multiload
sudo aptitude -y install atom cmake cmake-curses-gui
sudo aptitude -y install python-pip python-pip3
sudo aptitude -y install ipython
sudo -E pip3 install numpy
mkdir code
cd code
git config --global http.proxy http://proxy-chain.intel.com:911
git clone https://github.com/squeakus/bitsandbytes.git
cd bitsandbytes


#sudo apt-get install libgnome2-bin
#sudo apt-get -y install apache2
#sudo apt-get install mysql-server php5-mysql

cd bashrc/
sudo make install
cd ../
sudo chown jonathan:jonathan ~/.bashrc* 
echo "alias npython='ipython notebook --pylab inline'" >> ~/.bashrc.alias
echo "alias pyserver='python -m SimpleHTTPServer 8080 &'" >> ~/.bashrc.alias
echo "alias cmd='vi ~/code/bitsandbytes/bashrc/commands.txt'" >> ~/.bashrc.alias
echo "alias ppython='ipython --pylab'" >> ~/.bashrc.alias
echo "alias open='gvfs-open'" >> ~/.bashrc.alias
echo "alias python='python3'" >> ~/.bashrc.alias
#history | cut -c 8- > install.txt
