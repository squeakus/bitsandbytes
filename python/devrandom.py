rand = open('/dev/random','r')
for i in range(10):
    randVal =  rand.readline()
    print "Random Value:",str(randVal)
