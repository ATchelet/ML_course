# -*- coding: utf-8 -*-
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