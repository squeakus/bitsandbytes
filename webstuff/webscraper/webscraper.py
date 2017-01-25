import urllib2
from  BeautifulSoup import BeautifulSoup

#url="http://www.met.ie/latest/buoy.asp"
#url="http://irelandsweather.com/"
url="http://irelandsweather.com/modules/markers/markers.php"
page=urllib2.urlopen(url)
soup = BeautifulSoup(page.read())


print soup.prettify()
info = soup.findAll('a')
#info = soup.findAll('span')
#info = soup.findAll('span',{'class':'High'})
for data in info:
    print data
