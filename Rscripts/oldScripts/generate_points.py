import random

#myRandom  = Random()

random_points = open('random_points.dat', 'w')
for _ in range(200):
    x, y, z  = random.random(), random.random(), random.random() * 2
    print x,y,z
    line = str(x) + ' ' + str(y) + ' ' + str(z) + '\n'
    random_points.write(line)

