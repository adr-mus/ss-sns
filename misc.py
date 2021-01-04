import functools, math


def track(func):
    @functools.wraps(func)
    def inner(*args, **kwargs):
        print(f"Executing {func.__name__}...", end=" ")
        ret = func(*args, **kwargs)
        print("Done.")
        return ret
    return inner


def balancing_numbers(n):
    n_single = n
    n_pairs = n_single * (n_single - 1) // 2
    lcm = n_single * n_pairs // math.gcd(n_single, n_pairs)
    return lcm // n_single, lcm // n_pairs 