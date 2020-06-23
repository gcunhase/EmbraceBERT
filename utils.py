import os
import numpy as np
import matplotlib.pyplot as plt

__author__ = "Gwena Cunha"


def get_project_path():
    current_dir = os.path.abspath(os.path.dirname(__file__))
    return current_dir


def ensure_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def visualize_attention(attention_probs_img, cmap_name='plasma'):
    """ Brighter colors mean higher value. From low to high:
            blue, purple, red, orange, yellow

    :param attention_probs_img: Numpy array 128x128
    :return:
    """
    # Visualize attention_probs
    # Colormaps: https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
    #max_value = attention_probs.max()
    #min_value = attention_probs.min()
    # att_img = plt.matshow(attention_probs_img, cmap=plt.get_cmap('plasma'))
    att_img = plt.matshow(attention_probs_img, cmap=plt.get_cmap(cmap_name))  # 'Greys'))
    if attention_probs_img.shape[0] == 1:
        plt.yticks([])
    # plt.imshow(attention_probs_img)
    plt.show()
    return att_img
