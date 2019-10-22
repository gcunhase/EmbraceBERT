import os

__author__ = "Gwena Cunha"


def get_project_path():
    current_dir = os.path.abspath(os.path.dirname(__file__))
    return current_dir


def ensure_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
