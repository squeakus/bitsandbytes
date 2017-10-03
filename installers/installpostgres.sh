sudo aptitude -y install postgresql postgresql-contrib
sudo aptitude -y install gdal-bin
sudo aptitude -y install python3-pip
sudo -E pip3 install --upgrade pip
sudo -E pip3 install pyproj sqlalchemy flask
sudo add-apt-repository -y ppa:ubuntugis/ubuntugis-unstable
sudo aptitude update
sudo aptitude -y install postgis
sudo aptitude -y install pgadmin3

#all of this from the command line!
sudo -i -u postgres
createuser jonathan
createdb vola_db
exit
psql -d vola_db

cd volamap/flask/bin
sudo ln -s /usr/bin/python3 python
cd ../../app/static
chmod +x updatemap.sh
sudo cp ~/code/MvOLA/VOLA/chunker/vola.py /usr/local/bin/
sudo chmod +x /usr/local/bin/vola.py
./updatemap.sh
cd ../../
chmod +x run.py

###SECURITY#### set superuser password and ensure no remote access in bg_hba.conf
#/etc/postgresql/9.1/main/pg_hba.conf
#https://www.digitalocean.com/community/tutorials/how-to-secure-postgresql-on-an-ubuntu-vps

sudo -u postgres psql -d vola_db -c "CREATE EXTENSION postgis;"
sudo -u postgres psql -d vola_db -c "CREATE EXTENSION postgis_topology;"

#geoserver setup: http://docs.geoserver.org/stable/en/user/installation/linux.html
#download the binary: http://geoserver.org/download/
#install java 8

sudo apt-get install openjdk-8-jre

echo "export GEOSERVER_HOME=/usr/share/geoserver" >> ~/.profile
. ~/.profile
sudo chown -R USER_NAME /usr/share/geoserver/
#once all set up run geoserver/bin/startup.sh
http://localhost:8080/geoserver
DEFAULT = admin geoserver (don't forget to change!!)

# if No 'Access-Control-Allow-Origin' header is present on the requested then it needs to be enabled in jetty

