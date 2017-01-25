# WARNING LAT AND LON REVERSED

def gps_to_point(lon, lat):

    xcoord = (((180+lon) * (1 / 360.0))*32400) - 15230;
    ycoord = (((90-lat) * (1/ 180.0))*27000) - 5145;

    print xcoord, ycoord
    return xcoord, ycoord

coordfile = open('coords.txt','r')
pointsfile = open('points.js','w')
pointsfile.write("var point_list = [")

for line in coordfile:
    coord = eval(line)
    lat, lon = coord[0], coord[1]
    x,y = gps_to_point(lon, lat)
    pointsfile.write("["+str(x)+","+str(y)+"],\n")
pointsfile.write(']')
coordfile.close()
pointsfile.close()
