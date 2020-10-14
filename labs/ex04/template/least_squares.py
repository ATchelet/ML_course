# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np
from costs import compute_mse


def least_squares(y, tx):
    """calculate the least squares."""
    # ***************************************************
    # least squares:
    # returns mse, and optimal weights
    # ***************************************************
    ws = np.linalg.inv(tx.T@tx)@tx.T@y
    mse = compute_mse(y,tx,ws)
    return mse, ws