import cProfile
import functools


def profileit(func):
    @functools.wraps(func)  # <-- Changes here.
    def wrapper(*args, **kwargs):
        prof = cProfile.Profile()
        retval = prof.runcall(func, *args, **kwargs)
        print(prof.print_stats())
        # prof.dump_stats("profile.profile")
        return retval

    return wrapper
