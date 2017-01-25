import sys

#f = open('./workfile.txt', 'w')
#for arg in sys.argv: 
#    f.write(arg+'\n')
#f.close

#defining a function
def print_file(fileName): 
    f2= open(fileName,'r')
    for line in f2:
        print line.rstrip()

#dealing with arrrays
#x =[1,2,3,4,5,6,7]
#for element in x:
#    print element


# variables and conditionals
#x = 9
#if x < 10 :
#    print "x less than 10"
#    x =x+1
#elif x > 10:
#    print "x greater than 10"
    
if __name__ == "__main__":
    print_file(sys.argv[1])
