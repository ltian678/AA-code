import logging

def get_logger(logfile, name='rumourlogger', level=logging.DEBUG):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    format='%(asctime)s %(levelname)s %(message)s'

    #write to logfile
    fileHandler = logging.FileHandler(logfile)
    fileHandler.setFormatter(logging.Formatter(format))
    logger.addHandler(fileHandler)

    #print out log to console
    conHandler = logging.StreamHandler()
    conHandler.setFormatter(logging.Formatter(format))
    logger.addHandler(conHandler)
    logger.propagate = False
    return logger


def close(logger):
    #close logger
    logger.handlers = []
