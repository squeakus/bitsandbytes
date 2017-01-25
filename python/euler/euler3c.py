def factorize(to_factor):
    """Use trial division to factorize to_factor and return all the resulting
    factors."""
    factors = []
    divisor = 2
    while (divisor < to_factor):
        
        if not to_factor % divisor:
            print "original divisor ",divisor," and original factor ",to_factor
            to_factor /= divisor
            factors.append(divisor)
            # Note we don't bump the divisor here; if we did, we'd have
            # non-prime factors.
        elif divisor == 2:
            divisor += 1
        else:
            # Trivial optimization: skip even numbers that aren't 2.
            divisor += 2
    if not to_factor % divisor:
        # Don't forget the last factor
        factors.append(to_factor)
        print "last factor ",to_factor
    return factors

if __name__ == "__main__":
    print max(factorize(300))
