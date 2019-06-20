# coding=utf-8
"""
Create JSON of directory structure.

Created on Aug 30, 2018

@author: zachary.burnett
"""

import functools
import json
import os


def get_directory_structure(root_dir: str) -> dict:
    """
    Creates a nested dictionary that represents the folder structure of rootdir

    :param root_dir: directory that will be the root of the output
    :return: dictionary of folder structure
    """

    output_dict = {}
    root_dir = root_dir.rstrip(os.sep)
    start = root_dir.rfind(os.sep) + 1
    for path, dirs, files in os.walk(root_dir):
        folders = path[start:].split(os.sep)
        subdir = dict.fromkeys(files)
        parent = functools.reduce(dict.get, folders[:-1], output_dict)
        parent[folders[-1]] = subdir
    return output_dict


def dir_structure_to_json(input_dir: str, json_path: str):
    """
    Write directory structure to JSON file.

    :param input_dir: root directory
    :param json_path: output JSON
    """

    output_data = get_directory_structure(input_dir)

    with open(json_path, 'w') as json_file:
        json.dump(output_data, json_file)


if __name__ == '__main__':
    import sys

    sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir))

    from PyOFS import DATA_DIR

    dir_structure_to_json(os.path.join(DATA_DIR, 'output'), os.path.join(DATA_DIR, 'reference', 'files.json'))
