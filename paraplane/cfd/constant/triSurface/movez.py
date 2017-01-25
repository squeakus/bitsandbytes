import sys

if sys.argv[1] == None:
    print "pass in either an .stl or .obj file"
filename = sys.argv[1]

if sys.argv[1].endswith('.stl'):
    vertsymbol = 'vertex'
elif sys.argv[1].endswith('.obj'):
    vertsymbol = 'v'
else:
    print "not a valid file extension"
    exit()

    
stlfile = open(filename,'r')
outfile = open('plane2.stl','w')
vertcount = 0
for line in stlfile:
    tmpline = line.strip()
    
    if tmpline.startswith(vertsymbol):
        vertcount += 1
        info = tmpline.split(' ')
        z = float(info[3])
	z += 3
        newline = str(info[0]) + " " + str(info[1]) + " " + str(info[2]) + " " + str(z) + "\n"
        outfile.write(newline)
        print "*****************"
        print "before", line
        print "after", newline
    else:
	outfile.write(line)
outfile.close()
    
print vertcount, "vertices"
