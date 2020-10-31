import functools


def track(func):
    @functools.wraps(func)
    def inner(*args, **kwargs):
        print(f"Executing {func.__name__}...", end=" ")
        ret = func(*args, **kwargs)
        print("Done.")
        return ret
    return inner
