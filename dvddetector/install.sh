#!/bin/bash
# prerequisites: username exclusive


#get all the required packages
sudo apt-get -y install aptitude
sudo aptitude -y update
sudo aptitude -y install subversion mencoder vlc python-gudev python-tk libdvdread4 dconf-tools nfs-kernel-server vim lsdvd

#create the folders
sudo mkdir ~/dvddetector /export /export/videos /opt/udev-sh

#copy the files
sudo cp ./dvddetector.py ~/dvddetector

#fix libdvdcss
sudo /usr/share/doc/libdvdread4/install-css.sh

#turn off automount
dconf write /org/gnome/desktop/media-handling/automount false

#create the udev rules
echo KERNEL==\"sr[0-9]\", ACTION==\"change\", RUN+=\"/opt/udev-sh/sr-cd_dvd.sh\" | sudo tee /etc/udev/rules.d/10-cddvd-change.rules
echo -e '#!/bin/bash \necho "rule working" > /tmp/rule.txt\n/usr/bin/python '$HOME'/dvddetector/dvddetector.py' | sudo tee /opt/udev-sh/sr-cd_dvd.sh
sudo chmod +x /opt/udev-sh/sr-cd_dvd.sh

#mount the NFS point
sudo mount --bind $HOME/Videos/ /export/videos
sudo echo $HOME/Videos /export/videos   none    bind  0  0 | sudo tee -a /etc/fstab
sudo echo "/export 192.168.1.0/24(rw,fsid=0,insecure,no_subtree_check,async)" | sudo tee -a /etc/exports
sudo echo "/export/videos 192.168.1.0/24(rw,nohide,insecure,no_subtree_check,async)" | sudo tee -a /etc/exports
sudo /etc/init.d/nfs-kernel-server restart

