
# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Kyle Matoba <kmatoba@idiap.ch>
# SPDX-License-Identifier: BSD-3-Clause

import datetime as dt
import os
import getpass
import logging
import time


def _get_logging_fmt() -> str:
    logging_fmt = '%(asctime)s %(message)s'
    return logging_fmt


def _get_formatter() -> logging.Formatter:
    logging_fmt = _get_logging_fmt()
    formatter = logging.Formatter(logging_fmt)
    formatter.converter = time.gmtime
    return formatter


def _get_filename_str() -> str:
    filename_str = f"{dt.datetime.utcnow():%Y%m%d%H%M%S}.log"
    return filename_str


def get_standard_streamhandler():
    # https://docs.python.org/3/library/logging.html#logrecord-attributes
    formatter = _get_formatter()
    standard_streamhandler = logging.StreamHandler()
    standard_streamhandler.setFormatter(formatter)
    standard_streamhandler.setLevel(logging.INFO)
    return standard_streamhandler


def get_standard_filehandler(log_dir_str: str):
    filename_str = _get_filename_str()
    fullfilename_str = os.path.join(log_dir_str, filename_str)

    standard_filehandler = logging.FileHandler(fullfilename_str)
    standard_filehandler.setLevel(logging.DEBUG)
    return standard_filehandler


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(get_standard_streamhandler())
