import numpy as np

# Least Squares
def least_squares(y, tx):
    '''
    A method that calculates the optimal weights for x  to predict y using least squares method.

    usage: w, loss = least_squares(y, tx)

    input:
    -y  - output labels vector [Nx1]
    -tx - input features matrix [NxD]
    output:
    -w      - optimal weights [1xD]
    -loss   - overall distance of prediction from true label [scalar]
    '''
    w = np.linalg.inv(tx.transpose()@tx)@tx.transpose()@y #calculation of w* = (X^T.X).X^T.y
    e = y-tx@w #calculation of error
    loss = 0.5*e@e/len(y) #calculation of loss (MSE)
    return w, loss

# Regularized logistic regression using gradient descent or SGD
def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma, mode='GD'):
    '''
    A method that calculates the optimal weights for x  to predict y.

    usage: w, loss = least_squares(y, tx)

    input:
    -y          - output labels vector [Nx1]
    -tx         - input features matrix [NxD]
    -initial_w  - initial weights values [1xD]
    -max_iter   - maximal number of iterations [scalar]
    -gamma      - regularization parameter [scalar]
    -mode       - determines if the method uses gradient descent 'GD' or stochastic gradient descent 'SGD' ['GD' or 'SGD']
    output:
    -w      - optimal weights [1xD]
    -loss   - overall distance of prediction from true label [scalar]
    '''
    if not any(mode=='GD', mode=='SGD'):
        raise UnsupportedMode

    
    return w, loss