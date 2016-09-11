import math
import numpy as np
import scipy


def kernel(r, kernel_name='optimal'):
    if (kernel_name == 'optimal'):
        return 3./4*(1-r**2) if abs(r) <= 1 else 0
    if kernel_name == 'triangular':
        return 1 - abs(r) if abs(r) <= 1 else 0
    if kernel_name == 'gauss':
        return (2*math.pi)**(-1./2)*math.exp(-1./2*r**2)
    if kernel_name == 'rectangular':
        return 1/2 if abs(r) <= 1 else 0
    if kernel_name == 'quartic':
        return 15./16 * (1 - r**2)**2 if abs(r) <= 1 else 0
    return 0


class Window(object):
    def __init__(self):
        pass

    def fit(self, X_test, X, y, k, metric, kernel_name):
        self.X = X
        self.y = y
        self.X_test = X_test
        self.metric = metric
        self.kernel_name = kernel_name
        self.k = k
        return self

    def predict(self):
        dists = np.zeros((self.X_test.shape[0], self.X.shape[0]))
        if self.metric == 'l2':
            train_diag = np.sum(self.X**2, axis=1)
            test_diag = np.sum(self.X_test**2, axis=1)
            train_test = np.dot(self.X, self.X_test.transpose())
            dists = (train_test * (-2) + test_diag).transpose() + train_diag
            dists = dists**(1./2)
        elif self.metric == 'cosine':
            train_diag = np.sum(self.X**2, axis=1)
            test_diag = np.sum(self.X_test**2, axis=1)
            train_test = np.dot(self.X, self.X_test.transpose())
            train_diag = train_diag ** (1./2)
            test_diag = test_diag ** (1./2)
            dists = (train_test / test_diag).transpose() / train_diag.transpose()
            dists = np.ones(dists.shape) - dists
        elif self.metric == 'l1':
            for i in range(self.X_test.shape[0]):
                dists[i] = np.sum(abs(self.X - self.X_test[i]), axis=1)
        elif self.metric == 'l3':
            for i in range(self.X_test.shape[0]):
                dists[i] = np.sum(abs(self.X - self.X_test[i])**3, axis=1)**1./3

        predictions = []

        for i in xrange(self.X_test.shape[0]):
            neighbors = np.argsort(dists[i])[:self.k+1]
            kth = neighbors[self.k]
            neighbors = neighbors[:self.k]
            classes = {}
            for index in neighbors:
                r = 0
                r_k = 0
                if self.metric == 'l2':
                    r = scipy.linalg.norm(self.X_test[i]-self.X[index], 2)
                    r_k = scipy.linalg.norm(self.X_test[i]-self.X[kth], 2)
                elif self.metric == 'cosine':
                    r = 1 - np.dot(self.X_test[i],
                                   self.X[index])/(np.linalg.norm(self.X_test[i]) *
                                                   np.linalg.norm(self.X[index]))
                    r_k = 1 - np.dot(self.X_test[i],
                                     self.X[kth])/(np.linalg.norm(self.X_test[i]) *
                                                   np.linalg.norm(self.X[kth]))
                elif self.metric == 'l1':
                    r = scipy.linalg.norm(self.X_test[i]-self.X[index], 1)
                    r_k = scipy.linalg.norm(self.X_test[i]-self.X[kth], 1)
                elif self.metric == 'l3':
                    r = scipy.linalg.norm(self.X_test[i]-self.X[index], 3)
                    r_k = scipy.linalg.norm(self.X_test[i]-self.X[kth], 3)
                if self.y[index] in classes:
                    classes[self.y[index]] += kernel(r/r_k, self.kernel_name)
                else:
                    classes[self.y[index]] = kernel(r/r_k, self.kernel_name)
            prediction = max(classes.items(), key=lambda x: (x[1], x[0]))[0]
            predictions.append(prediction)
        return predictions
