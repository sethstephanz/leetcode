def sieve(self, range_min, range_max):
    """
    Efficient implementation of Sieve of Eratosthenes
    """
    is_prime = [True] * (range_max + 1)
    is_prime[0] = is_prime[1] = False

    for p in range(2, int(range_max ** 0.5) + 1):
        if is_prime[p]:
            for multiple in range(p * p, range_max + 1, p):
                is_prime[multiple] = False

    primes = [i for i, prime in enumerate(is_prime) if prime and i >= range_min]
    return primes
