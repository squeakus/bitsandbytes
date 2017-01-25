import struct

with open('IMG_1416.sift', 'rb') as f:
    content = f.read()
    #read the header
    offset = 20
    #name, version, npoint, notsure,dimension =
    print(struct.unpack('cccc cccc iii', content[:offset])) # int is 4 bytes
    name, version, npoints, notsure,dimension = struct.unpack('iiiii', content[:offset])
    print "name, version, npoint, notsure,dimension"

    #DONT ASK ME WHY BUT THEY REVERSE THE X AND Y
    #descriptor location and orientation
    for point in range(npoints):
        #print "point:", point, "offset:", offset
        y,x, color, scale, orientation = struct.unpack('fffff', content[offset:offset+20]) #float is 4 bytes
        offset += 20
        if point < 4:
            print x,y, scale, orientation

    #descriptor info
    for point in range(3):
        #print "point:", point, "offset:", offset
        print(struct.unpack('B'*128, content[offset:offset+128]))
        offset += 128
    #
    #print(struct.unpack('fffff', content[168:17]))
