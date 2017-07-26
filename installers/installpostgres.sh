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
