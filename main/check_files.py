# coding=utf-8
"""
Check existence of files.

Created on Apr 10, 2019

@author: zachary.burnett
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

from main import write_json


def check_files(input_dir: str) -> dict:
    missing_files = {}

    structure = write_json.get_directory_structure(input_dir)

    return missing_files


if __name__ == '__main__':
    from PyOFS import DATA_DIR

    missing_files = check_files(os.path.join(DATA_DIR, 'output'))
