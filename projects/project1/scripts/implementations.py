import numpy as np

########################    
### Helper functions ###
########################

### Useful function to implement SGD method ###
def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

### Mean square error calculation ###
def MSE(y, tx, w):
    """
    A method that calculates the mean square error (MSE).

    usage: loss = MSE(y, tx, w)

    input:
    -y  - output labels vector [Nx1]
    -tx - input features matrix [NxD]
    -w  - weights vector [Dx1]
    output:
    -loss   - distance of prediction from true label [scalar]
    """
    e = y-tx@w #calculation of error
    loss = 0.5*e@e/len(y) #calculation of loss (MSE)
    return loss

    ### Mean square error calculation ###
def logistic_loss(y, tx, w):
    """
    A method that calculates the loss of the log-likelihood loss.

    usage: loss = logistic_loss(y, tx, w)

    input:
    -y  - output labels vector [Nx1]
    -tx - input features matrix [NxD]
    -w  - weights vector [Dx1]
    output:
    -loss   - distance of prediction from true label [scalar]
    """
    # calculate logistic regression loss
    return np.sum(np.log(1+np.exp(-y*(tx@w))))
            
########################    
### Implementations  ###
########################

### Linear regression using gradient descent ###
def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    '''
    A method that calculates the optimal weights for x  to predict y by linear regression using gradient descent.

    usage: w, loss = least_squares_GD(y, tx, initial_w, max_iters, gamma)

    input:
    -y          - output labels vector [Nx1]
    -tx         - input features matrix [NxD]
    -initial_w  - initial weight values [Dx1]
    -max_iters  - number of iterations to perform [scalar]
    -gamma      - step size parameter [scalar]
    output:
    -w      - optimal weights [1xD]
    -loss   - distance of prediction from true label [scalar]
    '''
    
    w = initial_w
    
    for n_iter in range(max_iters):
        # compute gradient
        # tx is the x input matrix with the augmented 1 column at the beginning for the w0 parameter as the offset at axis origins
        e = y - np.dot(tx, w)# e is the error vector e = y - f(x). NB there is a calculated error for each datapoint
        grad = -np.dot(tx.T, e)/ len(e)
        
        # update weights
        w = w - gamma*grad
    
    # calculate error
    loss = MSE(y, tx, w)
    
    return w, loss


### Linear regression using stochastic gradient descent ###
def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    '''
    A method that calculates the optimal weights for x  to predict y by linear regression using stochastic gradient descent.

    usage: w, loss = least_squares_SGD(y, tx, initial_w, max_iters, gamma)

    input:
    -y          - output labels vector [Nx1]
    -tx         - input features matrix [NxD]
    -initial_w  - initial weight values [Dx1]
    -max_iters  - number of iterations to perform [scalar]
    -gamma      - step size parameter [scalar]
    output:
    -w      - optimal weights [1xD]
    -loss   - distance of prediction from true label [scalar]
    '''
    
    w = initial_w
    
    for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size=1, num_batches=max_iters):
        # compute gradient
        # tx is the x input matrix with the augmented 1 column at the beginning for the w0 parameter as the offset at axis origins
        e=minibatch_y-minibatch_tx@w
        grad=-np.transpose(minibatch_tx)@e/len(e)
        
        # update weights
        w = w - gamma*grad
    
    # calculate error
    loss = MSE(y, tx, w)
    
    return w, loss
    
    
### Least squares regression using normal equations ###
def least_squares(y, tx):
    '''
    A method that calculates the optimal weights for x  to predict y using least squares regression using normal equations.

    usage: w, loss = least_squares(y, tx)

    input:
    -y  - output labels vector [Nx1]
    -tx - input features matrix [NxD]
    output:
    -w      - optimal weights [1xD]
    -loss   - overall distance of prediction from true label [scalar]
    '''
    # update weights
    w = np.linalg.inv(tx.T@tx)@tx.T@y #calculation of w* = (X^T.X).X^T.y
   
    # calculate error
    loss = MSE(y, tx, w)
    
    return w, loss
    
    
### Ridge regression using normal equations ###
def ridge_regression(y, tx, lambda_):
    '''
    A method that calculates the optimal weights for x  to predict y using ridge regression using normal equations.
    
    usage: w, loss = ridge_regression(y, tx, lambda_)
    
    input:
    -y          - output labels vector [Nx1]
    -tx         - input features matrix [NxD]
    -lambda_    - regularizaition parameter [scalar]
    output:
    -w      - optimal weights [1xD]
    -loss   - overall distance of prediction from true label [scalar]
    '''    
    # There are two methods for solving the algebric system (both tested and working)
    # We want to solve the system Aw_star = b with b = X_T*y 
    
    A = np.dot(tx.T, tx) + 2*len(y)*lambda_*np.identity(len(tx.T))
    b = np.dot(tx.T, y)
    
    # 1. Manual method with inverse
    # w = np.dot(np.linalg.inv(A), b)

    # 2. With np solving method, solves x in Ax=b, here we have XˆT*X*w_star = XˆT*y => More robust method because works even if not inversible.
    w = np.linalg.solve(A, b)
    
    # calculate error
    loss = MSE(y, tx, w)
    
    return w, loss
    
    
### Logistic regression using gradient descent or SGD ###
def logistic_regression(y, tx, initial_w, max_iters, gamma, mode='SGD'):
    '''
    A method that calculates the optimal weights for x  to predict y in {0,1} using logistic regression using gradient descent or SGD.
    
    usage: w, loss = logistic_regression(y, tx, initial_w, max_iters, gamma)
    
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
    # Validate mode
    if not any([mode=='GD', mode=='SGD']):
        raise UnsupportedMode
        
    w = initial_w
    
    if mode=='SGD': # Stochastic Gradient Descent
        # cycle related to batches (in the case of SGD we only have one batch)
        for n_iter in range(max_iters):
            for y_batch, tx_batch in batch_iter(y, tx, batch_size=1, num_batches=1):
                # calculate gradient
                grad=tx_batch.T.dot(np.exp(tx_batch.dot(w))/(1+np.exp(tx_batch.dot(w)))-y_batch)
                # update weights
                w=w-gamma*grad
                
    else: # mode='GD' gradient descent method
        for n_iter in range(max_iters):
            # calculate gradient
            grad=tx.T.dot(np.exp(tx.dot(w))/(1+np.exp(tx.dot(w)))-y)
            # update weights
            w=w-gamma*grad
            
    # calculate error
    loss=np.sum(np.log(1+np.exp(tx.dot(w)))-y*tx.dot(w))
    
    return w, loss
    
    
# Regularized logistic regression using gradient descent or SGD
def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma, mode='SGD'):
    '''
    A method that calculates the optimal weights for x to predict y in {-1,1} using regularized logistic regression using gradient descent or SGD

    usage: w, loss = reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma)

    input:
    -y          - output labels vector [Nx1]
    -tx         - input features matrix [NxD]
    -lambda_    - regularizaition parameter [scalar]
    -initial_w  - initial weights values [1xD]
    -max_iter   - maximal number of iterations [scalar]
    -gamma      - regularization parameter [scalar]
    -mode       - determines if the method uses gradient descent 'GD' or stochastic gradient descent 'SGD' ['GD' or 'SGD']
    output:
    -w      - optimal weights [1xD]
    -loss   - overall distance of prediction from true label [scalar]
    '''
    # Validate mode
    if not any([mode=='GD', mode=='SGD']):
        raise UnsupportedMode
    
    w = initial_w
    if mode=='SGD': # Stochastic Gradient Descent
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=1, num_batches=max_iters):
            # calculate gradient
            # grad = tx_batch.T@((1/(1+np.exp(-tx_batch@w)))-y_batch) + lambda_*w
            grad = tx_batch.T@(-y_batch/(1+np.exp(y_batch*(tx_batch.dot(w))))) + lambda_*w
            # update weights
            w = w + gamma*grad 

    else: # Gradient Descent
        for n_iter in range(max_iters):
            # calculate gradient
            # grad = tx.T@((1/(1+np.exp(-tx@w)))-y) + lambda_*w
            grad = tx.T@(-y/(1+np.exp(y*(tx.dot(w))))) + lambda_*w
            # update weights
            w = w + gamma*grad

    # calculate error
    loss = np.sum(np.log(1+np.exp(-y*(tx@w)))) + 0.5*lambda_*w.dot(w)
    
    return w, loss
    
# Logistic regression with Newton's Method
def logistic_regression_newton(y, tx, initial_w, max_iters, gamma):
    '''
    A method that calculates the optimal weights for x to predict y using logistic regression using Newton's method

    usage: w, loss = logistic_regression_newton(y, tx, initial_w, max_iters, gamma)

    input:
    -y          - output labels vector [Nx1]
    -tx         - input features matrix [NxD]
    -initial_w  - initial weights values [1xD]
    -max_iter   - maximal number of iterations [scalar]
    -gamma      - regularization parameter [scalar]
    output:
    -w      - optimal weights [1xD]
    -loss   - overall distance of prediction from true label [scalar]
    '''
    w = initial_w

    # iterate to optimize weights
    for n_iter in range(max_iters):
        # calculate gradient
        grad=tx.T.dot(np.exp(tx.dot(w))/(1+np.exp(tx.dot(w)))-y)
        # calculate Hessian matrix - H=X.T@S@X , S = diag(sigma(X.T@w)*[1-sigma(X.T@w)])
        H = tx.T@(np.diagflat((1/(1+np.exp(-tx.dot(w))))*(1-(1/(1+np.exp(-tx.dot(w)))))))@tx
        # update weights
        # w -= gamma*(np.linalg.inv(H)@grad)
        w = np.linalg.solve(H, w-gamma*grad)

    # calculate error
    loss = np.sum(np.log(1+np.exp(tx@w))-y*(tx@w))
    
    return w, loss
    
# Regularized logistic regression using gradient descent or SGD
def reg_logistic_regression_newton(y, tx, lambda_, initial_w, max_iters, gamma):
    '''
    A method that calculates the optimal weights for x to predict y using regularized logistic regression using Newton's method

    usage: w, loss = reg_logistic_regression_newton(y, tx, lambda_, initial_w, max_iters, gamma)

    input:
    -y          - output labels vector [Nx1]
    -tx         - input features matrix [NxD]
    -lambda_    - regularizaition parameter [scalar]
    -initial_w  - initial weights values [1xD]
    -max_iter   - maximal number of iterations [scalar]
    -gamma      - regularization parameter [scalar]
    output:
    -w      - optimal weights [1xD]
    -loss   - overall distance of prediction from true label [scalar]
    '''
    w = initial_w

    for n_iter in range(max_iters):
        # calculate gradient
        grad=tx.T.dot(np.exp(tx.dot(w))/(1+np.exp(tx.dot(w)))-y) + lambda_*w
        # calculate Hessian matrix - H=X.T@S@X , S = diag(sigma(X.T@w)*[1-sigma(X.T@w)])
        H = tx.T@(np.diagflat((1/(1+np.exp(-tx.dot(w))))*(1-(1/(1+np.exp(-tx.dot(w)))))))@tx
        # update weights
        # w -= gamma*(np.linalg.inv(H)@grad)
        w = np.linalg.solve(H, w-gamma*grad)

    # calculate error
    loss = np.sum(np.log(1+np.exp(tx@w))-y*(tx@w)) + 0.5*lambda_*w.dot(w)
    
    return w, loss
    
# Logistic regression with Newton's Method
def logistic_regression_newton(y, tx, initial_w, max_iters, gamma):
    '''
    A method that calculates the optimal weights for x to predict y using logistic regression using Newton's method

    usage: w, loss = logistic_regression_newton(y, tx, initial_w, max_iters, gamma)

    input:
    -y          - output labels vector [Nx1]
    -tx         - input features matrix [NxD]
    -initial_w  - initial weights values [1xD]
    -max_iter   - maximal number of iterations [scalar]
    -gamma      - regularization parameter [scalar]
    output:
    -w      - optimal weights [1xD]
    -loss   - overall distance of prediction from true label [scalar]
    '''
    w = initial_w

    # iterate to optimize weights
    for n_iter in range(max_iters):
        # calculate gradient
        grad=tx.T.dot(np.exp(tx.dot(w))/(1+np.exp(tx.dot(w)))-y)
        # calculate Hessian matrix - H=X.T@S@X , S = diag(sigma(X.T@w)*[1-sigma(X.T@w)])
        H = tx.T@(np.diagflat((1/(1+np.exp(-tx.dot(w))))*(1-(1/(1+np.exp(-tx.dot(w)))))))@tx
        # update weights
        # w -= gamma*(np.linalg.inv(H)@grad)
        w = np.linalg.solve(H, w-gamma*grad)

    # calculate error
    loss = np.sum(np.log(1+np.exp(tx@w))-y*(tx@w))
    
    return w, loss
    
# Regularized logistic regression using gradient descent or SGD
def reg_logistic_regression_newton(y, tx, lambda_, initial_w, max_iters, gamma):
    '''
    A method that calculates the optimal weights for x to predict y using regularized logistic regression using Newton's method

    usage: w, loss = reg_logistic_regression_newton(y, tx, lambda_, initial_w, max_iters, gamma)

    input:
    -y          - output labels vector [Nx1]
    -tx         - input features matrix [NxD]
    -lambda_    - regularizaition parameter [scalar]
    -initial_w  - initial weights values [1xD]
    -max_iter   - maximal number of iterations [scalar]
    -gamma      - regularization parameter [scalar]
    output:
    -w      - optimal weights [1xD]
    -loss   - overall distance of prediction from true label [scalar]
    '''
    w = initial_w

    for n_iter in range(max_iters):
        # calculate gradient
        grad=tx.T.dot(np.exp(tx.dot(w))/(1+np.exp(tx.dot(w)))-y) + lambda_*w
        # calculate Hessian matrix - H=X.T@S@X , S = diag(sigma(X.T@w)*[1-sigma(X.T@w)])
        H = tx.T@(np.diagflat((1/(1+np.exp(-tx.dot(w))))*(1-(1/(1+np.exp(-tx.dot(w)))))))@tx
        # update weights
        # w -= gamma*(np.linalg.inv(H)@grad)
        w = np.linalg.solve(H, w-gamma*grad)

    # calculate error
    loss = np.sum(np.log(1+np.exp(tx@w))-y*(tx@w)) + 0.5*lambda_*w.dot(w)
    
    return w, loss

