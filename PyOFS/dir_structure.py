# coding=utf-8
"""
Create JSON of directory structure.

Created on Aug 30, 2018

@author: zachary.burnett
"""

import functools
import json
import os

DATA_DIR = os.environ['OFS_DATA']


def get_directory_structure(rootdir):
    """
    Creates a nested dictionary that represents the folder structure of rootdir

    :param rootdir: directory that will be the root of the output
    :return: dictionary of folder structure
    """

    output_dict = {}
    rootdir = rootdir.rstrip(os.sep)
    start = rootdir.rfind(os.sep) + 1
    for path, dirs, files in os.walk(rootdir):
        folders = path[start:].split(os.sep)
        subdir = dict.fromkeys(files)
        parent = functools.reduce(dict.get, folders[:-1], output_dict)
        parent[folders[-1]] = subdir
    return output_dict


def dir_structure_to_json(input_dir, json_path):
    """
    Write directory structure to JSON file.

    :param input_dir: root directory
    :param json_path: output JSON
    """

    output_data = get_directory_structure(input_dir)

    with open(json_path, 'w') as json_file:
        json.dump(output_data, json_file)


if __name__ == '__main__':
    json_path = os.path.join(DATA_DIR, r'reference\model_dates.json')
    input_dir = os.path.join(DATA_DIR, 'output')

    dir_structure_to_json(input_dir, json_path)
