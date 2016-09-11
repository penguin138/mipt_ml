import numpy as np
from math import exp, log
# from random import shuffle


def softmax_loss_naive(W, X, y, lambda_):
    """
    Softmax loss function, naive implementation (with loops)
    Inputs:
    - W: C x D array of weights
    - X: D x N array of data. Data are D-dimensional columns
    - y: 1-dimensional array of length N with labels 0...K-1, for K classes
    - reg: (float) regularization strength
    Returns:
    a tuple of:
    - loss as single float
    - gradient with respect to weights W, an array of same size as W
    """
    # Initialize the loss and gradient to zero.
    dW = np.zeros(W.shape)
    loss = 0
    all_scores = np.dot(W, X)
    all_scores = all_scores.transpose()
    X = X.transpose()
    data_loss = 0
    for i in range(X.shape[0]):
        scores = all_scores[i]
        true_class = y[i]
        true_class_score = scores[true_class]
        current_loss = 0
        for score in scores:
            current_loss += exp(score)
        for k in range(W.shape[0]):
            dW[k] += (exp(scores[k]) / current_loss) * X[i]
            if k == true_class:
                dW[k] -= X[i]
        current_loss = log(current_loss) - true_class_score
        data_loss += current_loss
    data_loss /= X.shape[0]
    regularization_loss = lambda_ * np.sum(W ** 2)
    loss = data_loss + regularization_loss
    dW /= X.shape[0]
    dW += 2 * lambda_ * W
    return loss, dW


def softmax_loss_vectorized(W, X, y, lambda_):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros(W.shape)
    all_scores = np.dot(W, X)
    all_scores_exp = np.exp(all_scores)
    data_loss = (np.sum(np.log(np.sum(all_scores_exp, axis=0))) -
                 np.sum(all_scores[y, np.arange(y.shape[0])]))
    data_loss /= X.shape[1]
    regularization_loss = lambda_ * np.sum(W ** 2)
    loss = data_loss + regularization_loss
    new_scores = all_scores_exp/all_scores_exp.sum(axis=0)
    new_scores[y, np.arange(y.shape[0])] += -1
    dW += np.dot(new_scores, X.T)
    dW /= X.shape[1]
    dW += 2 * lambda_ * W
    return loss, dW
