import sys

outfile = open("damn.txt", 'w')
outfile.write("holy shit:", sys.argv[1])
outfile.close()
print "ha!"
