import random

random.seed(10)
for i in range(5):
	print random.randint(1,100)


myRandGen = random.Random(10)

print "break"
for i in range(5):
	print random.randint(1,100)

print "myrng"
for i in range(5):
	print myRandGen.randint(1,100)
	
