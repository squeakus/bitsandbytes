import json

infile = open('output1.csv','r')
stationnames = ['FENIAN STREET', 'CITY QUAY', 'FITZWILLIAM SQUARE EAST', 'BROOKFIELD ROAD', 'EMMET ROAD', 'ROTHE ABBEY', 'KING STREET NORTH', 'GREEK STREET', 'WESTERN WAY', 'CHARLEMONT PLACE', 'PARKGATE STREET', 'HIGH STREET', 'HEUSTON BRIDGE (SOUTH)', 'PORTOBELLO HARBOUR', 'ECCLES STREET EAST', 'MERRION SQUARE EAST', 'SMITHFIELD', 'PORTOBELLO ROAD', 'CUSTOM HOUSE QUAY', 'MERRION SQUARE WEST', 'HARDWICKE PLACE', "ST. STEPHEN'S GREEN EAST", 'MOLESWORTH STREET', 'UPPER SHERRARD STREET', 'MOUNTJOY SQUARE WEST', 'LIME STREET', 'EXCHEQUER STREET', 'ORMOND QUAY UPPER', 'NEW CENTRAL BANK', 'DAME STREET', 'PARNELL SQUARE NORTH', 'DEVERELL PLACE', 'EARLSFORT TERRACE', 'BOLTON STREET', 'JAMES STREET EAST', 'ECCLES STREET', 'CATHAL BRUGHA STREET', 'STRAND STREET GREAT', 'FITZWILLIAM SQUARE WEST', 'FOWNES STREET UPPER', 'HERBERT STREET', 'PARNELL STREET', "ST. STEPHEN'S GREEN SOUTH", 'PEARSE STREET', 'EXCISE WALK', 'TALBOT STREET', 'CUSTOM HOUSE', 'GEORGES QUAY', 'LEINSTER STREET SOUTH', 'WILTON TERRACE', 'TOWNSEND STREET', 'JERVIS STREET', 'HEUSTON STATION (CENTRAL)', 'SMITHFIELD NORTH', 'KEVIN STREET', 'NEWMAN HOUSE', 'SANDWITH STREET', 'MOUNT STREET LOWER', 'HANOVER QUAY', 'CHRISTCHURCH PLACE', "SIR PATRICK DUN'S", 'GUILD STREET', 'YORK STREET WEST', 'HARCOURT TERRACE', 'YORK STREET EAST', 'ST JAMES HOSPITAL (LUAS)', 'CLONMEL STREET', 'JOHN STREET WEST', 'JAMES STREET', 'MARKET STREET SOUTH', 'MATER HOSPITAL', 'CONVENTION CENTRE', "PRINCES STREET / O'CONNELL STREET", 'WOLFE TONE STREET', 'OLIVER BOND STREET', 'BARROW STREET', 'HERBERT PLACE', 'GRAND CANAL DOCK', 'DENMARK STREET GREAT', 'FRANCIS STREET', 'NORTH CIRCULAR ROAD', 'THE POINT', 'KILMAINHAM LANE', 'BLESSINGTON STREET', 'ROYAL HOSPITAL', 'COLLINS BARRACKS MUSEUM', 'BLACKHALL PLACE', 'KILMAINHAM GAOL', 'HEUSTON STATION (CAR PARK)', 'BENSON STREET', 'FREDERICK STREET SOUTH', 'SOUTH DOCK ROAD', 'ST. JAMES HOSPITAL (CENTRAL)', 'HEUSTON BRIDGE (NORTH)', 'MOUNT BROWN', 'CHATHAM STREET', 'GRANTHAM STREET', 'GRATTAN STREET', 'HARDWICKE STREET', 'HATCH STREET', 'GOLDEN LANE']
timelist = []
laststation = 'GOLDEN LANE'
stations = {}

for line in infile:
    line = line.split(',')
    stationname = line[5]
    time = line[7]
    time = time.split('.')[0]
    bikes = line[1]
    free = line[2]

    if stationname != laststation:
        if stationname != "name":
            stations[stationname] = (bikes,free)

    else:
        stations[stationname] = (bikes,free)
        if len(timelist) > 0:
            prevtime = timelist[-1][0]
        else:
            prevtime = 0
        if time != prevtime:
            timelist.append((time, stations))
        else:
            print "duplicate time recorded!"
        stations = {}

outfile = open('stationdata.json','w')

#compact
outfile.write(json.dumps(timelist, separators=(',', ':')))

#pretty
#outfile.write(json.dumps(timelist, sort_keys=True,
#                        indent=4, separators=(',', ': ')))

outfile.close()
