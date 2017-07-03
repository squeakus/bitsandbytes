#!/bin/bash
sudo add-apt-repository -y ppa:xorg-edgers/ppa
sudo add-apt-repository ppa:webupd8team/atom
sudo apt-get update
sudo apt-get -y install aptitude git subversion vim compizconfig-settings-manager compiz-plugins compiz-plugins-extra
sudo aptitude -y install nodejs npm
sudo ln -s /usr/bin/nodejs /usr/bin/node
sudo npm install -g -y http-server
sudo aptitude -y install openssh-server
sudo aptitude -y install fail2ban
sudo aptitude -y install indicator-multiload
#sudo apt-get -y install apache2
#sudo apt-get install mysql-server php5-mysql
sudo aptitude -y install ipython
#sudo apt-get install libgnome2-bin

cd bashrc/
sudo make install
cd ../

echo "alias npython='ipython notebook --pylab inline'" >> ~/.bashrc.alias
echo "alias pyserver='python -m SimpleHTTPServer 8080 &'" >> ~/.bashrc.alias
echo "alias cmd='vi ~/code/bitsandbytes/bashrc/commands.txt'" >> ~/.bashrc.alias
echo "alias ppython='ipython --pylab'" >> ~/.bashrc.alias
echo "alias open='gvfs-open'" >> ~/.bashrc.alias
echo "alias python='python3'" >> ~/.bashrc.alias
#history | cut -c 8- > install.txt
