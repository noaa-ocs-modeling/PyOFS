# coding=utf-8
"""
Create JSON of directory structure.

Created on Aug 30, 2018

@author: zachary.burnett
"""

import functools
import json
import os
from os import PathLike
from pathlib import Path
import sys

sys.path.append(Path(__file__).resolve().parent.parent.parent)

from PyOFS import DATA_DIRECTORY, get_logger

LOGGER = get_logger('PyOFS.JSON')


def get_directory_structure(root_dir: PathLike) -> dict:
    """
    Creates a nested dictionary that represents the folder structure of rootdir

    :param root_dir: directory that will be the root of the output
    :return: dictionary of folder structure
    """

    if not isinstance(root_dir, str):
        root_dir = str(root_dir)

    output_dict = {}
    root_dir = root_dir.rstrip(os.sep)
    start = root_dir.rfind(os.sep) + 1
    for path, dirs, files in os.walk(root_dir):
        folders = path[start:].split(os.sep)
        subdir = dict.fromkeys(files)
        parent = functools.reduce(dict.get, folders[:-1], output_dict)
        parent[folders[-1]] = subdir
    return output_dict


def dir_structure_to_json(input_dir: PathLike, json_path: PathLike):
    """
    Write directory structure to JSON file.

    :param input_dir: root directory
    :param json_path: output JSON
    """

    output_data = get_directory_structure(input_dir)

    with open(json_path, 'w') as json_file:
        json.dump(output_data, json_file)


if __name__ == '__main__':
    dir_structure_to_json(DATA_DIRECTORY / 'output', DATA_DIRECTORY / 'reference' / 'files.json')
