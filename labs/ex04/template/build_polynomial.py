# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    # ***************************************************
    # polynomial basis function:
    # this function should return the matrix formed
    # by applying the polynomial basis to the input data
    # ***************************************************
    poly_x = np.zeros((len(x), degree+1))
    for j in range (degree+1):
        poly_x[:,j] = x**j
    return poly_x
