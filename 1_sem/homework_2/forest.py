from node import Node
import numpy as np
from scipy import stats


def bagging(X, y):
    size = len(X)
    sample_indices = np.random.choice(size, size, replace=True)
    sample_X, sample_y = X[sample_indices], y[sample_indices]
    return sample_X, sample_y


class Forest(object):
    def __init__(self, num_trees, max_depth=16, leaf_node_size=4,
                 criterion='gini'):
        self.num_trees = num_trees
        self.max_depth = max_depth
        self.leaf_node_size = leaf_node_size
        self.criterion = criterion

    def fit(self, X_train, y_train):
        '''
        Creation of trees using bagging
        '''
        self.trees = []
        for i in xrange(self.num_trees):
            tree = Node(self.max_depth, self.leaf_node_size, self.criterion)
            sample_X, sample_y = bagging(X_train, y_train)
            tree.fit(sample_X, sample_y)
            self.trees.append(tree)

    def predict(self, X_test):
        '''
        Prediction of the label using generated trees.
        '''
        predictions = []
        for tree in self.trees:
            y_pred = tree.predict(X_test)
            predictions.append(y_pred)
        predictions = np.array(predictions)
        return stats.mode(predictions)[0]
