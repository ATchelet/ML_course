import numpy as np

# Useful function to implement SGD method (to move in a "helper" file)
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


            
####### Least_square GD method ######################### 1
"""Least squares using gradient descent"""

def compute_gradient(y, tx, w):
    """Compute the gradient.
    
    Return
    ------
    """
    
    N = y.shape[0] # N is the number of datapoints
    
    # tx is the x input matrix with the augmented 1 column at the beginning for the w0 parameter as the offset at axis origins
    e = y - np.dot(tx, w)# e is the error vector e = y - f(x). NB there is a calculated error for each datapoint
    
    gradient = -np.dot(tx.T, e)/ N
    
    return gradient

def gradient_descent(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    
    w = initial_w
    
    for n_iter in range(max_iters):

        #print(w)
        # compute gradient and loss
        loss = compute_loss(y, tx, w)
        #print(loss)
        gradient = compute_gradient(y, tx, w)

        # ***************************************************
        #raise NotImplementedError
        # ***************************************************
        # INSERT YOUR CODE HERE
        w = w - gamma*gradient
        # TODO: update w by gradient
        # ***************************************************
        #raise NotImplementedError
        # store w and loss
        ws.append(w)
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses, ws

############# least_squares using SGD ################################# 2
#compute loss function using least square (Mean square error)
def compute_loss(y, tx, w):
    '''
    input:
    -y          - output labels vector [Nx1]
    -tx         - input features matrix [NxD]
    -w          -  weights values [Dx1]
    '''
    e=y-tx@w
    loss=0.5*e.T.dot(e)/y.shape[0]
    return loss

#compute SGD of MSE (it's the same as the gradient)    
def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    '''
    input:
    -y          - output labels vector [Nx1]
    -tx         - input features matrix [NxD]
    -w          -  weights values [Dx1]
    '''
    e=y-tx@w
    grad=-np.transpose(tx)@e/len(e)
    return grad

            
#Implementing gradient descent method
def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent algorithm."""
    w = initial_w
    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=1, num_batches=1):
            grad=compute_stoch_gradient(y_batch, tx_batch, w)
            loss=least_squares_loss(y, tx, w)
            w=w-gamma*grad
    return w, loss

######## Least Squares ############ 3
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


######### ridge_regression ################ 4
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


################## logistic regression ##################### 5
def logistic_function(x,w):
    l=np.exp(x.dot(w))/(1+np.exp(x.dot(w)))
    return l

def loss_logistic(y,tx,w):
    '''This function to minimaze obtained from Maximum likelihood'''
    loss=np.sum(np.log(1+np.exp(tx.dot(w)))-y*tx.dot(w))
    return loss

def compute_gradient_logistic(y, tx, w):
    """Compute gradient of loss logistic respect to w """
    grad=tx.T.dot(np.exp(tx.dot(w))/(1+np.exp(tx.dot(w)))-y)
    return grad

def compute_stoch_gradient_logistic(y, x, w):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    grad=x.T.dot(np.exp(x.dot(w))/(1+np.exp(x.dot(w)))-y)
    return grad

def logistic_regression(y, tx, initial_w, max_iters, gamma, mode='SGD'):
    """Logistic regression using stochastic gradient descent algorithm."""
    # Validate mode
    if not any(mode=='GD', mode=='SGD'):
        raise UnsupportedMode
    w = initial_w
    if mode=='SGD': # Stochastic Gradient Descent
        # cycle related to batches (in the case of SGD we only have one batch)
        for n_iter in range(max_iters):
            for y_batch, tx_batch in batch_iter(y, tx, batch_size=1, num_batches=1):
                grad=compute_stoch_gradient_logistic(y_batch, tx_batch, w)
                loss=loss_logistic(y, tx, w)
                w=w-gamma*grad
    else: # mode='GD' gradient descent method
        for n_iter in range(max_iters):
            grad=compute_gradient_logistic(y, tx, w)
            loss=loss_logistic(y, tx, w)
            w=w-gamma*grad
    return loss, w

# prediction for logistic regression (if probability is more than 0.5 we choose 1 otherwise we choose -1)
def prediction(x,w):
    if logistic_function(x,w)>=0.5:
        y=1
    else:
        y=-1
    return y


########### Regularized logistic regression using gradient descent or SGD ############## 6
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

