import logging


tools_global_logger_judger = None
def tools_get_logger(name:str='test'):
    global tools_global_logger_judger
    if not tools_global_logger_judger:
        fmt = "{asctime} {name:8s} {msg}"
        root = logging.getLogger()
        root.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        bf = logging.Formatter(fmt, style='{', datefmt='%y-%m-%d %H:%M:%S')
        handler.setFormatter(bf)
        root.addHandler(handler)
    return logging.getLogger(name)