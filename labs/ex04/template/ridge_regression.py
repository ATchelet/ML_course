# -*- coding: utf-8 -*-
"""Exercise 3.

Ridge Regression
"""

import numpy as np


def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    # ***************************************************
    # ridge regression:
    # ***************************************************
    lambda_t = 2*len(y)*lambda_
    ws = np.linalg.inv(tx.T@tx + lambda_t*np.identity(tx.shape[1]))@tx.T@y
    e = y - tx@ws
    mse = 0.5 * e@e / len(y) + lambda_*ws@ws
    return mse, ws
