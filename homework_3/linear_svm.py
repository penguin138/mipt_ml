import numpy as np
# from random import shuffle


def svm_loss_naive(W, X, y, lambda_):
    """
    Structured SVM loss function, naive implementation (with loops)
    Inputs:
    - W: C x D array of weights
    - X: D x N array of data. Data are D-dimensional columns
    - y: 1-dimensional array of length N with labels 0...K-1, for K classes
    - reg: (float) regularization strength
    Returns:
    a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    dW = np.zeros(W.shape)  # initialize the gradient as zero
    data_loss = 0
    all_scores = np.dot(W, X)
    all_scores = all_scores.transpose()
    X = X.transpose()
    for i in range(X.shape[0]):
        x = X[i]  # i-th element
        scores = all_scores[i]
        true_class = y[i]
        true_class_score = scores[true_class]
        current_loss = 0
        for j in range(scores.shape[0]):
            max_ = max(0, scores[j] - true_class_score + 1)
            if max_ > 0:
                if j != true_class:
                    current_loss += max_
                dW[j] += x
                dW[true_class] -= x
        data_loss += current_loss
    data_loss /= X.shape[0]
    regularization_loss = 0
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            regularization_loss += W[i][j] ** 2
            dW[i][j] /= X.shape[0]
            dW[i][j] += 2*lambda_*W[i][j]
    regularization_loss *= lambda_
    loss = data_loss + regularization_loss
    return loss, dW


def svm_loss_vectorized(W, X, y, lambda_):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    all_scores = np.dot(W, X)
    all_scores = all_scores - all_scores[y, np.arange(all_scores.shape[1])] + 1
    data_loss = np.sum(np.maximum(all_scores, 0)) - all_scores.shape[1]
    data_loss /= all_scores.shape[1]
    regularization_loss = lambda_ * np.sum(W ** 2)
    loss = data_loss + regularization_loss
    ##########################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    ##########################################################################
    
    dW /= X.shape[0]
    dW += 2*lambda_ * W
    ##########################################################################
    #                             END OF YOUR CODE                              #
    ##########################################################################
    return loss, dW
