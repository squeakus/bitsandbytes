mesh= open('xxx.1.mesh','r')
for line in mesh:
    result = line.split()
    if len(result) > 3:
        #print "coords: ",result
        x,y,z = float(result[0]),float(result[1]),float(result[2])
        if x > 30:
            print "bigX:",x
        if y > 15:
            print "bigY:",y
        if z > 15:
            print "bigZ:",y

