# import datetime
import time


def time_comp(fun):
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = fun(*args, **kwargs)
        tf = time.perf_counter()
        dt = tf - t0
        if kwargs.get("verbose"):
            print("Time elapsed: ", "{:.2f}".format(dt) + "s")

        return result

    return wrapper


def time_comp_cls(fun):
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = fun(*args, **kwargs)
        tf = time.perf_counter()
        dt = tf - t0
        if getattr(args[0], "verbose", False):
            print("Time elapsed: ", "{:.2f}".format(dt) + "s")

        return result

    return wrapper
