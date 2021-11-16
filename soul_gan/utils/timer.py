# import datetime
import time


def time_comp(fun):
    def wrapper(*args, **kwargs):
        # date = str(datetime.datetime.now())[0:10]
        # params = args[0]
        t0 = time.perf_counter()
        model = fun(*args, **kwargs)

        tf = time.perf_counter()
        dt = tf - t0

        # unpack
        # params = model['params']
        # x = model['x']
        # w = model['weight']
        # feature = model['feature']

        print("Time elapsed: ", "{:.2f}".format(dt) + "s")
        # fd_name = date + '_' + params['name']
        # save_end(x, w, params, fd_name, dt, feature)
        # save_model(model, fd_name)

        return model

    return wrapper
