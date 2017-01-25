import urllib2, re, json
from math import *
from  BeautifulSoup import BeautifulSoup

def gps_to_point(lat, lon):
    xcoord = (((180+lon) * (1 / 360.0))*32400) - 15230;
    ycoord = (((90-lat) * (1/ 180.0))*27000) - 5145;
    return [xcoord, ycoord]

def download_weather(url):
    names, coords, directs, speeds = [], [], [], []
    max_lat, min_lat, max_lon, min_lon = -1000,1000,-10000,1000
    page=urllib2.urlopen(url)
    for idx,line in enumerate(page):
        parts = line.split(';')
        for part in parts:
            #print part, '\n'
            if part.startswith('var point = new GLatLng'):
                coord = part.lstrip('var point = new GLatLng')
                tupcoord = eval(coord)
                lat, lon = tupcoord[0], tupcoord[1]
                coords.append((lat,lon))

        m = re.search('table(.+?)table', line)
        if m:
            found = m.group(1)
            found = '<table' + found +'table>'
            found = found.replace('\\','')
            soup = BeautifulSoup(found)
            info = soup.findAll('tr',{'class':'wind_row'})
            name = soup.findAll('a')
            names.append(name[0].text)
            for data in info:
                direction = str(data.find('img'))
                direction = direction.rstrip('.png" />')
                direction = direction.lstrip('<img src="images/wind/')
                directs.append(direction)
                n = re.search('Wind:(.+?)km', str(data))
                if n:
                    speed = n.group(1)
                    speeds.append(speed)

    # parse info into a dictionary array
    data_list = []
    for i in range(len(names)):
        if not names[i].startswith("Buoy"):
            lat, lon = coords[i][0], coords[i][1]
            data_list.append(dict(name=names[i], 
                                  gps=list(coords[i]), 
                                  direct=directs[i], 
                                  speed=speeds[i]))

    # find max lon and lat
    for data in data_list:
        lat, lon = data['gps'][0], data['gps'][1]    
        if lat > max_lat: max_lat = lat
        if lat < min_lat: min_lat = lat
        if lon > max_lon: max_lon = lon
        if lon < min_lon: min_lon = lon
            
    for data in data_list:
        lat, lon = data['gps'][0], data['gps'][1] 

        if lat == max_lat or lat == min_lat:
            print "extreme lat", data['name'], lat, lon
        if lon == max_lon or lon == min_lon:
            print "extreme lon", data['name'], lat, lon


    winddir = {'N':0,
               'NNE':22.5,
               'NE':45,
               'ENE':67.5,
               'E':90,
               'ESE':112.5,
               'SE':135,
               'SSE':157.5,
               'S':180,
               'SSW':202.5,
               'SW':225,
               'WSW':247.5,
               'W':270,
               'WNW':292.5,
               'NW':315,
               'NNW':337.5,
               '---':0,
               '--':0,
               '-':0}

    print "data points:", len(data_list)
    jsfile = open('weather.json', 'w')

    jsfile.write('[')
    for data in data_list:
        data["name"] = str(data["name"])
        data["mapcoord"] = gps_to_point(data["gps"][0],data["gps"][1])
        data["windangle"] = winddir[data["direct"]]
        if data == data_list[-1]:
            jsfile.write(json.dumps(data)+'\n')
        else:
            jsfile.write(json.dumps(data)+',\n')
    jsfile.write(']')
    jsfile.close()

        
def main():
    url="http://irelandsweather.com/modules/markers/markers.php"
    download_weather(url)

if __name__=='__main__':
    main()
    
