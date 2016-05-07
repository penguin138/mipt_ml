#! /usr/bin/env python3
import numpy as np
from linear_svm import svm_loss_vectorized
from softmax import softmax_loss_vectorized


class LinearClassifier:

    def __init__(self):
        self.W = None

    def train(self, X, y, learning_rate=1e-3, lambda_=1e-5, num_iters=100,
              batch_size=200, verbose=False):
        """
        Train this linear classifier using stochastic gradient descent.

        Inputs:
        - X: D x N array of training data. Each training point is a D-dimensional
             column.
        - y: 1-dimensional array of length N with labels 0...K-1, for K classes.
        - learning_rate: (float) learning rate for optimization.
        - lambda_: (float) regularization strength.
        - num_iters: (integer) number of steps to take when optimizing
        - batch_size: (integer) number of training examples to use at each step.
        - verbose: (boolean) If true, print progress during optimization.

        Outputs:
        A list containing the value of the loss function at each training iteration.
        """
        dim, num_train = X.shape
        # assume y takes values 0...K-1 where K is number of classes
        num_classes = np.max(y) + 1
        if self.W is None:
            # lazily initialize W
            self.W = np.random.randn(num_classes, dim) * 0.001
        best_W = np.zeros(self.W.shape)
        best_loss = 0
        # Run stochastic gradient descent to optimize W
        loss_history = []  # to plot losses
        for it in range(num_iters):
            permutation = np.random.choice(np.arange(X.shape[1]), batch_size)
            X_batch = X[:, permutation]
            y_batch = y[permutation]
            loss, gradient = self.loss(X_batch, y_batch, lambda_)
            self.W = self.W - learning_rate * gradient
            if loss_history == [] or loss < min(loss_history):
                best_W = self.W
                best_loss = loss
            loss_history.append(loss)
            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f, best_loss %f' % (it, num_iters, loss, best_loss))
        self.W = best_W
        return loss_history

    def predict(self, X):
        """
        Use the trained weights of this linear classifier to predict labels for
        data points.

        Inputs:
        - X: D x N array of training data. Each column is a D-dimensional point.

        Returns:
        - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
          array of length N, and each element is an integer giving the predicted
          class.
        """
        y_pred = np.argmax(np.dot(self.W, X), axis=0)
        return y_pred

    def loss(self, X_batch, y_batch, reg):
        """
        Compute the loss function and its derivative.
        Subclasses will override this.

        Inputs:
        - X_batch: D x N array of data; each column is a data point.
        - y_batch: 1-dimensional array of length N with labels 0...K-1, for K classes.
        - reg: (float) regularization strength.

        Returns: A tuple containing:
        - loss as a single float
        - gradient with respect to self.W; an array of the same shape as W
        """
        pass


class LinearSVM(LinearClassifier):
    """ A subclass that uses the Multiclass SVM loss function """

    def loss(self, X_batch, y_batch, reg):
        return svm_loss_vectorized(self.W, X_batch, y_batch, reg)


class Softmax(LinearClassifier):
    """ A subclass that uses the Softmax + Cross-entropy loss function """

    def loss(self, X_batch, y_batch, reg):
        return softmax_loss_vectorized(self.W, X_batch, y_batch, reg)

    def predict_probabilities(self, X):
        all_scores = np.dot(self.W, X)
        all_scores_exp = np.exp(all_scores)
        probs = (all_scores_exp/all_scores_exp.sum(axis=0)).T
        return probs
