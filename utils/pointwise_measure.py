"""
.. module:: pointwise_measure
.. moduleauthor:: Rui Xin Lee <rlee7@jaguarlandrover.com>

"""

import numpy as np


def pointwise(array_a, array_b):
    a_len = np.min([array_a.shape[0], array_a.shape[0]])
    array_a = array_a[:a_len]
    array_b = array_b[:a_len]
    distance = np.sum(np.abs(array_a - array_b))
    return distance
