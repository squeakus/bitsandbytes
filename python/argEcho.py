import sys

for arg in sys.argv: 
    print arg
if len(sys.argv) > 1:
   print "first arg:",sys.argv[1]
else:
   print "no arguments passed"
