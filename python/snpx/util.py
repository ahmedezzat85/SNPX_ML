from __future__ import absolute_import

import os

def snpx_create_dir(dir_path):
    """ Create a directory.
    """
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    if not os.path.isdir(dir_path):
        raise ValueError("Cannot Create Directory %s", dir_path)

class DictToAttrs(object):
    def __init__(self, d):
        self.__dict__ = d