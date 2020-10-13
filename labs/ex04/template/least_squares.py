# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np


def least_squares(y, tx):
    """calculate the least squares."""
    # ***************************************************
    # least squares:
    # returns mse, and optimal weights
    # ***************************************************
    ws = np.linalg.inv(tx.transpose()@tx)@tx.transpose()@y
    e = y - tx@ws
    mse = 0.5 * e@e / len(y)
    return mse, ws