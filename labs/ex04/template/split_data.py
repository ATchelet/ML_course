# -*- coding: utf-8 -*-
"""Exercise 3.

Split the dataset based on the given ratio.
"""


import numpy as np


def split_data(x, y, ratio, seed=1):
    """split the dataset based on the split ratio."""
    # set seed
    np.random.seed(seed)
    # ***************************************************
    # split the data based on the given ratio: 
    # ***************************************************
    idx = int(ratio*len(y))
    x_tr = x[:idx]
    y_tr = y[:idx]
    x_te = x[idx:]
    y_te = y[idx:]
    return x_tr, y_tr, x_te, y_te
