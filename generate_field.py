# generate a small field with a large order-2 subgroup
# wouldn't use this for much more than 2^20, slow
# p=65537, g=3, subgroup order=2^16
# p=7340033, g=3, subgroup order=2^20
# p=104857601, g=3, subgroup order=2^22
ORDER_SUBGROUP=22

def is_prime(n: int) -> bool:
    """Check if the number n is a prime number."""
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def gen_field_with_subgroup_order(n: int) -> int:
    """obtain a field with a subgroup of order 2**n"""
    n_ = 2**n
    p = n_ + 1
    # multiple = 1
    while not is_prime(p):
        # multiple += 1
        p += n_

    # print(multiple)
    return p

def is_primitive_root(candidate: int, prime: int) -> bool:
    """Check if candidate is a primitive root of prime"""
    s = set()
    acc = 1
    for i in range(1, p//2 + 2):
        acc *= candidate 
        acc %= prime
        if acc in s:
            return False
        s.add(acc)
    return True

def find_generator(prime: int) -> int:
    """take a prime field modulus, and find a generator"""
        
    for candidate in range(2, prime):
        if is_primitive_root(candidate, prime):
            return candidate

if __name__ == "__main__":
    p = gen_field_with_subgroup_order(ORDER_SUBGROUP)
    g = find_generator(p)
    assert(g**(2**(ORDER_SUBGROUP-1)) % p != 1)

    print(f"p={p}, g={g}, subgroup order=2^{ORDER_SUBGROUP}")