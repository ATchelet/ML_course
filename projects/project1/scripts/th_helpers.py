# -*- coding: utf-8 -*-
""" Many useful functions created during the labs """

import numpy as np

def split_data(x, y, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8 
    you will have 80% of your data set dedicated to training 
    and the rest dedicated to testing
    """    
    # set seed
    np.random.seed(seed)
    # ***************************************************
    # INSERT YOUR CODE HERE
    # split the data based on the given ratio: TODO
    
    if ratio<0 or ratio>1:
        raise NameError("Ratio is out of [0 1] range")

    nb_data_pts = len(y)
    nb_train_pts = int(np.rint(nb_data_pts*ratio))
    
    train_true_false = np.full(nb_data_pts, False)

    train_pts_indexes = np.random.choice(np.arange(start=0, stop=nb_data_pts), size=nb_train_pts, replace=False) # high is actually one above the max possible integer the function might return
    train_true_false[train_pts_indexes] = True
    
    train_x = x[train_true_false]
    train_y = y[train_true_false]
    
    test_x = x[~train_true_false]
    test_y = y[~train_true_false]
    
    return train_x, train_y, test_x, test_y
    
    # ***************************************************
    #raise NotImplementedError
    
# THEO test my code : 
#split_data(x=np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]]).T, y=np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19]).T, ratio=0.7)

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree.
    
    Output
    ------
    phi_x : matrix formed of augmented features, 
        from x = [x1, x2, x3...].T it returns
        phi_x = [[1, x1, x1ˆ2, x1ˆ3, ..., x1ˆdegree],
                 [1, x2, x2ˆ2, x2ˆ3, ..., x2^degree],
                 ...]
    """
    # polynomial basis function:
    # this function returns the matrix formed
    # by applying the polynomial basis to the input data
    
    # First get N and D, the number of datapoints and features
    if x.ndim == 1:
        raise NameError('This function only works for multi feature data')
    N = x.shape[0]
    D = x.shape[1]
        
    phi_x = np.empty(shape=(N, 1+D*degree))
    phi_x[:,0] = 1
    
    for degree_i in range(degree):
            phi_x[:, (1+degree_i*D):(1+(1+degree_i)*D)] = np.power(x, degree_i+1)
            
    return phi_x
# THEO TEST CODE
# import numpy as np
# from th_helpers import build_poly
# var_test_x = np.array([[0, 2],
#               [3, 4],
#              [5, 6]])
# build_poly(var_test_x,3)
# => should return
# array([[  1.,   0.,   2.,   0.,   4.,   0.,   8.],
#        [  1.,   3.,   4.,   9.,  16.,  27.,  64.],
#        [  1.,   5.,   6.,  25.,  36., 125., 216.]])




def compute_loss(y, tx, w):
    """Calculate the mean squared error loss.
    
    Return
    ------
    mse : mean squared error, noramlize to RMSE with np.sqrt(2*mse) 
    """
    # ***************************************************
    # INSERT YOUR CODE HERE
    mse = np.dot( (y-np.dot(tx, w)).T, y-np.dot(tx, w))/ (2*y.shape[0])

    return mse
    # TODO: compute loss by MSE / MAE
    # ***************************************************
    #raise NotImplementedError
    

def compute_rmse(y, tx, w):
    """Calculate the root mean squared error (standarized loss).
    
    Return
    ------
    rmse : root mean squared error.
    """
    mse = compute_loss(y, tx, w)
    rmse = np.sqrt(2*mse)

    return rmse
   