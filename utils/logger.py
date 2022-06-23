#!/usr/bin/python3

import logging
import os

logging_level = {'debug': logging.DEBUG,
                 'info': logging.INFO,
                 'warning': logging.WARNING,
                 'error': logging.ERROR,
                 'critical': logging.CRITICAL}


def debug(msg):
    logging.debug(msg)
    print('DEBUG: ', msg)


def info(msg):
    logging.info(msg)
    print('INFO: ', msg)


def warning(msg):
    logging.warning(msg)
    print('WARNING: ', msg)


def error(msg):
    logging.error(msg)
    print('ERROR: ', msg)


def fatal(msg):
    logging.critical(msg)
    print('FATAL: ', msg)


def logger_func(args):
    logger = logging.getLogger(args.log_name)
    logger.setLevel(logging_level[args.log_level])
    log_dir = os.path.dirname(args.log_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s : %(message)s', datefmt='%Y/%m/%d %H:%M:%S')

    # File Handler
    fh = logging.FileHandler(args.log_file, mode='w')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # stream Handler
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    sh.setLevel(logging_level[args.log_level])
    logger.addHandler(sh)
    # args.logger = logger
    return logger
