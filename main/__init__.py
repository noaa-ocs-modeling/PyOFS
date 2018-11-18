# coding=utf-8

import os

try:
    DATA_DIR = os.environ['OFS_DATA']
except KeyError:
    DATA_DIR = r"B:\Workspaces\Models\OFS_Data"
