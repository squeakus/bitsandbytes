from BeautifulSoup import BeautifulSoup
import re

page = open('out2.txt','r')

for idx,line in enumerate(page):
    parts = line.split(';')
    for part in parts:
        #print part, '\n'
        if part.startswith('var point = new GLatLng'):
            print "\n", part.lstrip('var point = new GLatLng')

    m = re.search('table(.+?)table', line)
    if m:
        found = m.group(1)
        found = '<table' + found +'table>'
        found = found.replace('\\','')
        soup = BeautifulSoup(found)
        info = soup.findAll('tr',{'class':'wind_row'})
        name = soup.findAll('a')
        print name[0].text
        for data in info:
            direction = str(data.find('img'))
            direction = direction.rstrip('.png" />')
            direction = direction.lstrip('<img src="images/wind/')
            print direction
            n = re.search('Wind:(.+?)km', str(data))
            if n:
                speed = n.group(1)
                print speed
