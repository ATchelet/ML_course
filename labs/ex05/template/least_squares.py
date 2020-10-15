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
    ws = np.linalg.inv(tx.T@tx)@tx.T@y
    return ws


#     e = y - tx.dot(w)
#     mse = e.dot(e) / (2 * len(e))