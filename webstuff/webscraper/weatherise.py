infile = open('weather.dat','r')
#outfile = open('weather.js', 'w')
#outfile.write('var weatherdata = [')

for line in infile:
    data = eval(line)
    data['name'] = str(data['name'])
    print data['speed']
    #outfile.write(str(data)+',\n')

#outfile.write(']')
infile.close()
#outfile.close()
