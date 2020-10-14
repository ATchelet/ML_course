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
    w = np.linalg.inv(tx.T@tx)@tx.T@y #calculation of w* = (X^T.X).X^T.y
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
    # Validate mode
    if not any(mode=='GD', mode=='SGD'):
        raise UnsupportedMode
    
    w = initial_w
    if mode=='SGD': # Stochastic Gradient Descent
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=1, num_batches=max_iters):
            exp_Xw = np.exp(tx_batch.T@w)
            sigma = exp_Xw/(1+exp_Xw)
            grad = tx_batch.T@(sigma-y_batch) + lambda_*np.sum(w)
            w = w - gamma*grad

    else: # Gradient Descent
        for n_iter in range(max_iters):
            exp_Xw = np.exp(tx.T@w)
            sigma = exp_Xw/(1+exp_Xw)
            grad = tx.T@(sigma-y) + lambda_*np.sum(w)
            w = w - gamma*grad

    loss = np.sum(np.log(1+np.exp(tx.T@w))-y*tx.T@w) + 0.5*lambda_*w.dot(w)
    
    return w, loss

    import numpy as np
############# least_squares using SGD #################################
#compute loss function using least square
def least_square_loss(y, tx, w):
    e=y-tx@w
    loss=0.5*e.T.dot(e)/y.shape[0]
    return loss
    
def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    e=y-tx@w
    grad=-np.transpose(tx)@e
    return grad

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
            
#Implementing gradient descent method
def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent algorithm."""
    w = initial_w
    for y_batch, tx_batch in batch_iter(y, tx, batch_size=1, num_batches=max_iters):
        grad=compute_stoch_gradient(y_batch, tx_batch, w)
        loss=least_squares_loss(y, tx, w)
        w=w-gamma*grad
    return loss, w

################## logistic regression #####################
def logistic_function(x,w):
    l=np.exp(x.dot(w))/(1+np.exp(x.dot(w)))
    return l

def loss_logistic(y,tx,w):
    '''This function to minimaze obtained from Maximum likelihood'''
    loss=np.sum(np.log(1+np.exp(tx.dot(w)))-y*tx.dot(w))
    return loss

def compute_stoch_gradient_logistic(y, x, w):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    grad=x.T.dot(np.exp(x.dot(w))/(1+np.exp(x.dot(w)))-y)
    return grad

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using stochastic gradient descent algorithm."""
    w = initial_w
    # cycle related to batches (in the case of SGD we only have one batch)
    for y_batch, tx_batch in batch_iter(y, tx, batch_size=1, num_batches=max_iters):
        grad=compute_stoch_gradient_logistic(y_batch, tx_batch, w)
        loss=loss_logistic(y, tx, w)
        w=w-gamma*grad
    return loss, w
def prediction(x,w):
    if logistic_function(x,w)>=0.5:
        y=1
    else:
        y=-1
    return y
