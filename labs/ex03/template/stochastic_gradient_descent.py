# -*- coding: utf-8 -*-
"""Stochastic Gradient Descent"""

def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    # ***************************************************
    # implement stochastic gradient computation.It's same as the gradient descent.
    # ***************************************************
    e = y - tx@w
    loss = 0.5*e@e/len(y)
    grad = - e@tx/len(y)
    return loss, grad


def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    # ***************************************************
    # implement stochastic gradient descent.
    # ***************************************************
    ws = [initial_w]
    losses = []
    w = initial_w
    n_iter = 0
    for minibatch_y, minibatch_tx in batch_iter(y, tx, max_iters):
        # ***************************************************
        # compute gradient and loss
        # ***************************************************
        loss, grad = compute_gradient(minibatch_y, minibatch_tx, w)
        # ***************************************************
        # update w by gradient
        # ***************************************************
        w -= gamma*grad
        # store w and loss
        ws.append(w)
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
        n_iter += 1
    return losses, ws