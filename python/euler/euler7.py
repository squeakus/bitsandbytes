def iter_primes ():
    # handle trivial case
    yield 2
    val = 1
    primesq_pairs = []
    while True:
        curr = None
        while (curr is None):
            val += 2
            curr = val
            for prime, square in primesq_pairs:
                if (curr < square):
                    break
                if (curr % prime == 0):
                    curr = None
                    break
                primesq_pairs.append ((curr, curr**2))
                yield curr


primer_gen = iter_primes()
for x in xrange (10001):
    result = primer_gen.next()
print result
