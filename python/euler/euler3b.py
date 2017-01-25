target = 600851475143
FILE = open("primes.txt","r")
for num in FILE:
    num =  num.rstrip()
    if target % int(num) ==0:
        print "factor:",num
FILE.close()
