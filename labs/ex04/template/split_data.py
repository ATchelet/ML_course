# -*- coding: utf-8 -*-
"""Exercise 3.

Split the dataset based on the given ratio.
"""


import numpy as np


def split_data(x, y, ratio, seed=1):
    """split the dataset based on the split ratio."""
    # set seed
    np.random.seed(seed)
    i_shuffle = np.random.permutation(len(y))
    x_s = x[i_shuffle]
    y_s = y[i_shuffle]
    # ***************************************************
    # split the data based on the given ratio: 
    # ***************************************************
    idx = int(ratio*len(y))
    x_tr = x_s[:idx]
    y_tr = y_s[:idx]
    x_te = x_s[idx:]
    y_te = y_s[idx:]
    return x_tr, y_tr, x_te, y_te
