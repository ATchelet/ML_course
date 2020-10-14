# -*- coding: utf-8 -*-
"""Function used to compute the loss."""

import numpy as np

def ridge_regression(y, tx, lambda_):
    """Performs ridge regression.
    
    Parameters
    ----------
    in : y, tx, lambda_
    out : w
    
    Return
    ------
    w : computed wheights with ridge regression
    """
    # NB this is very similar code to least_sqares
    
    #There are two methods for solving the algebric system (both tested and working)
    # We want to solve the system Aw_star = b with b = X_T*y
    
    A = np.dot(tx.T, tx) + 2*len(y)*lambda_*np.identity(len(tx.T))
    b = np.dot(tx.T, y)
    
    # 1. Manual method with inverse
    # w = np.dot(np.linalg.inv(A), b)

    # 2. With np solving method, solves x in Ax=b, here we have XˆT*X*w_star = XˆT*y => More robust method because works even if not inversible.
    w = np.linalg.solve(A, b)
    
    return w

    
# THEOTEST code
# test_w = ridge_regression(
#     y = np.array([[10, 11, 12, 13, 16, 18, 19]]).T,
#     tx = np.array([[  1.,   0.,   0.,   0.],
#                   [  1.,   1.,   1.,   1.],
#                   [  1.,   2.,   4.,   8.],
#                   [  1.,   3.,   9.,  27.],
#                   [  1.,   6.,  36., 216.],
#                   [  1.,   8.,  64., 512.],
#                   [  1.,   9.,  81., 729.]]),  
#     lambda_ = 0
# )
# print("Ridge w =\n", test_w, '\n')