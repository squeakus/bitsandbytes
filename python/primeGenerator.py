#Takes in a number, and returns all primes between 2 and that number
def sieve(max):
    #Start with all of the numbers
    primes = range(2,max+1)
    #Start running through each number
    for i in primes:
        #Start with double the number, and
        j = 2
        #remove all multiples
        print "processing: ",i
        while i * j <= primes[-1]:
            #As long as the current multiple of the number
            #is less than than the last element in the list
            #If the multiple is in the list, take it out
            if i * j in primes:
                primes.remove(i*j)
                j += 1
	return primes


if __name__ == "__main__":
    print "hello"
    primeList = sieve(1000000)
    for num in primeList:
        print num
    print "done"
