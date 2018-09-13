import functools
import json
import os


def get_directory_structure(rootdir):
    """
    Creates a nested dictionary that represents the folder structure of rootdir
    """
    dir = {}
    rootdir = rootdir.rstrip(os.sep)
    start = rootdir.rfind(os.sep) + 1
    for path, dirs, files in os.walk(rootdir):
        folders = path[start:].split(os.sep)
        subdir = dict.fromkeys(files)
        parent = functools.reduce(dict.get, folders[:-1], dir)
        parent[folders[-1]] = subdir
    return dir


def populate_json(input_dir, json_path):
    output_data = get_directory_structure(input_dir)

    with open(json_path, 'w') as json_file:
        json.dump(output_data, json_file)


if __name__ == '__main__':
    data_dir = os.environ['OFS_DATA']
    json_path = os.path.join(data_dir, r'reference\model_dates.json')
    input_dir = os.path.join(data_dir, 'output')

    populate_json(input_dir, json_path)
