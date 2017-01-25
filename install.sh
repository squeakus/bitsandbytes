#!/bin/bash
sudo add-apt-repository ppa:xorg-edgers/ppa
sudo apt-get update
aptitude search python
sudo apt-get install aptitude git subversion vim compizconfig-settings-manager compiz-plugins compiz-plugins-extra
sudo aptitude install nodejs npm
sudo aptitude install openssh-server
sudo aptitude install fail2ban
sudo aptitude install indicator-multiload
sudo apt-get install apache2
sudo apt-get install mysql-server php5-mysql
sudo aptitude install ipython
sudo apt-get install libgnome2-bin

svn co http://ncra.ucd.ie/svn/Jonathan/programs/bashrc
cd bashrc/
sudo make install

alias npython="ipython notebook --pylab inline"
alias pyserver='python -m SimpleHTTPServer 8080 &'
alias cmd='vi ~/Jonathan/Documents/commands.txt'
alias ppython='ipython --pylab'
alias open="gvfs-open" 

#history | cut -c 8- > install.txt
