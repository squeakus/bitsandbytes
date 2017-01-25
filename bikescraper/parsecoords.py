import json
from pprint import pprint

with open('stations.json') as data_file:
    data = json.load(data_file)

coords = {}
for elem in data['stations']:
    name = elem['name']
    lat = elem['latitude']
    lng = elem['longitude']
    print str(name) + ": " + str(lat) + str(lng)

    coords[name] = {'lat':lat,'lng':lng}

outfile = open('stationcoords.json','w')

#compact
outfile.write(json.dumps(coords, separators=(',', ':')))
#pretty
# outfile.write(json.dumps(coords, sort_keys=True,
#                         indent=4, separators=(',', ': ')))
outfile.close()
