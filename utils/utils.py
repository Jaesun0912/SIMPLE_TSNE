import time
import sys


def dict_key_val_as_filename(dict_: dict) -> str:
    order_key = list(dict_.keys())
    order_key.sort()
    str_ = ''
    for key in order_key:
        str_ += f'_{key}_{dict_[key]}'
    return str_


def dict_key_val_as_string(dict_: dict) -> str:
    order_key = list(dict_.keys())
    order_key.sort()
    str_ = ''
    for idx, key in enumerate(order_key):
        if idx == 0:
            str_ += f'{key}: {dict_[key]}'
        else:
            str_ += f', {key}: {dict_[key]}'
    return str_


def flush_print(str_: str):
    print(str_)
    sys.stdout.flush()


def time_trace(func):
    '''
    Function wrapper for checking elapsed time
    '''
    def wrapper(*args, **kwargs):
        st = time.time()
        rt = func(*args, **kwargs)
        if rt is None or rt == 0:
            flush_print(f'### {func.__name__} elapsed time : {time.time()-st:.3f}s')
        return rt
    return wrapper