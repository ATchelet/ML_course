# -*- coding: utf-8 -*-
"""Function used to compute the loss."""

def compute_loss(y, tx, w, mode='mse'):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    # ***************************************************
    # compute loss by MSE / MAE
    # ***************************************************
    e = y - tx@w
    if mode == 'mse':
        return 0.5*e@e/len(y)
    if mode == 'mae':
        return np.mean(np.abs(e))
    