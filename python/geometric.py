import math
x = 1
y = 100

results = open('harmresult.dat', 'w')

for i in range(99):
    arith_mean = (x + y)/2
    smallest = min([x,y])
    geo_mean = round(math.sqrt(x * y),2)
    z = (1.0/x) + (1.0/y)
    harmonic = round(2.0 / z, 2)

    print "mean for",x,"and",y,"arith:",arith_mean, "geometric", geo_mean, "harmonic:", harmonic
    x += 1
    y -= 1
    results.write(str(x) + ' ' + str(harmonic) + '\n')
results.close()
