import numpy as np
import scipy.linalg
from sklearn.neighbors import KDTree


class MatrixBasedKNN(object):
    """ A kNN classifier with different metrics """

    def __init__(self, num_loops):
        self.dist_mt = None
        self.num_loops = num_loops

    def fit(self, X_train, y_train, metric):
        """ Only save training data. """

        self.X_train = X_train
        self.y_train = y_train
        self.metric = metric
        return self

    def predict(self, X_test, k=1):
        """
        Predict labels for test data using this classifier.

        Inputs:
        - X_test: A numpy array of shape (num_test, D) containing test data consisting
                of num_test samples each of dimension D.
        - k: The number of nearest neighbors that vote for the predicted labels.
        - num_loops: Determines which implementation to use to compute distances
        between training points and testing points.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
            test data, where y[i] is the predicted label for the test point X[i].
        """

        num_train_objects = self.X_train.shape[0]
        num_test_objects = X_test.shape[0]
        self.dist_mt = np.zeros((num_test_objects, num_train_objects))

        if self.num_loops == 2:  # only for l2 norm
            for i in xrange(num_test_objects):
                for j in xrange(num_train_objects):
                    self.dist_mt[i][j] = scipy.linalg.norm(X_test[i]-self.X_train[j], 2)

        if self.num_loops == 1:  # only for l2 norm
            for i in range(num_test_objects):
                self.dist_mt[i] = np.sum((self.X_train - X_test[i])**2, axis=1)**1./2

        if self.num_loops == 0:  # can be used with different metrics
            if self.metric == 'l2':
                train_diag = np.sum(self.X_train**2, axis=1)
                test_diag = np.sum(X_test**2, axis=1)
                train_test = np.dot(self.X_train, X_test.transpose())
                self.dist_mt = (train_test * (-2) + test_diag).transpose() + train_diag
                self.dist_mt = self.dist_mt**(1./2)
            elif self.metric == 'cosine':
                train_diag = np.sum(self.X_train**2, axis=1)
                test_diag = np.sum(X_test**2, axis=1)
                train_test = np.dot(self.X_train, X_test.transpose())
                train_diag = train_diag ** (1./2)
                test_diag = test_diag ** (1./2)
                self.dist_mt = (train_test / test_diag).transpose() / train_diag.transpose()
                self.dist_mt = np.ones(self.dist_mt.shape) - self.dist_mt
            elif self.metric == 'l1':
                for i in range(num_test_objects):
                    self.dist_mt[i] = np.sum(abs(self.X_train - X_test[i]), axis=1)
            elif self.metric == 'l3':
                for i in range(num_test_objects):
                    self.dist_mt[i] = np.sum(abs(self.X_train - X_test[i])**3, axis=1)**1./3
        return self.predict_labels(self.dist_mt, k=k)

    def predict_labels(self, dists, k=1):
        """
            Given a matrix of distances between test points and training points,
            predict a label for each test point.

            Inputs:
            - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
              gives the distance betwen the ith test point and the jth training point.

            Returns:
            - y: A numpy array of shape (num_test,) containing predicted labels for the
              test data, where y[i] is the predicted label for the test point X[i].
        """
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            # A list of length k storing the labels of the k nearest neighbors to
            # the ith test point.
            closest_y = []
            sorted_indices = np.argsort(dists[i])[:k]
            # print(sorted_indices)
            for j in range(k):
                closest_y.append(self.y_train[sorted_indices[j]])
            closest = {}
            for pred in closest_y:
                if pred in closest:
                    closest[pred] += 1
                else:
                    closest[pred] = 1
            y_pred[i] = max(closest.items(), key=lambda x: (x[1], x[0]))[0]
        return y_pred

    def getNeighbours(self, index):
        sorted_indices = np.argsort(self.dist_mt[index])
        return sorted_indices


class KDTreeBasedKNN(object):
    def __init__(self):
        pass

    def fit(self, X_train, y_train):
        """
            Build KDtree using
            http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KDTree.html
        """

        self.X_train = X_train
        self.y_train = y_train
        self.kd_tree = KDTree(self.X_train)
        return self

    def predict(self, X_test, k=1):
        """
            Make prediction using kdtree
            Return array of prediction labels
        """
        y_pred = np.zeros(X_test.shape[0])

        for i in range(X_test.shape[0]):
            dists, sorted_indices = self.kd_tree.query(X_test[i].reshape((1, X_test[i].shape[0])),
                                                       k=k)
            # print(sorted_indices)
            closest_y = []
            for j in range(k):
                closest_y.append(self.y_train[sorted_indices[0][j]])
            closest = {}
            for pred in closest_y:
                if pred in closest:
                    closest[pred] += 1
                else:
                    closest[pred] = 1
            y_pred[i] = max(closest.items(), key=lambda x: (x[1], x[0]))[0]
        return y_pred
