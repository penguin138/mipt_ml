from criteria import gini, twoing, entropy
import numpy as np


class Node(object):

    def __init__(self, max_depth, max_X_length,
                 criterion='gini', depth=1, node_id=0):
        self.right_child = None
        self.left_child = None
        self.tree_level = depth  # current tree level of node
        self.variables = None  # dict: var_number -> var_bounds
        self.max_tree_depth = max_depth
        self.max_X_length = max_X_length  # max size of X for node to be leaf
        self.root_node = (depth == 1)
        self.criterion = criterion
        self.node_id = node_id

    def fit(self, X, y):
        self.X = X  # train set
        self.y = y  # train set labels
        if self.stop_condition():
            self.leaf_node = True
            label_0 = 0
            label_1 = 0
            if y is not None:
                for label in y:
                    if label == 0:
                        label_0 += 1
                    else:  # label == 1
                        label_1 += 1
                self.answer = 0 if label_0 >= label_1 else 1
            else:
                self.answer = 0
            self.size_of_subtree = 1
        else:
            self.variables = {}
            for i in xrange(len(self.X[0])):
                self.variables[i] = np.unique(self.X.transpose()[i])
            self.leaf_node = False
            self.split()
            self.size_of_subtree = (1 + self.left_child.size_of_subtree +
                                    self.right_child.size_of_subtree)
        return self.size_of_subtree

    def impurity(self, left_y, right_y):
        if self.criterion == 'gini':
            length = float(len(self.y))
            left_coef = len(left_y)/length
            right_coef = len(right_y)/length
            return left_coef * gini(left_y) + right_coef * gini(right_y)
        elif self.criterion == 'twoing':
            twoing_value = twoing(left_y, right_y)
            if twoing_value == 0:
                return float('inf')
            else:
                return 1 / twoing_value
        elif self.criterion == 'entropy':
            length = float(len(self.y))
            left_coef = len(left_y)/length
            right_coef = len(right_y)/length
            return left_coef * entropy(left_y) + right_coef * entropy(right_y)
        else:
            raise ValueError("Unknown criterion!")

    def stop_condition(self):
        return (self.tree_level >= self.max_tree_depth or
                len(self.X) < self.max_X_length or
                len(np.unique(self.y)) <= 1 or
                gini(self.y) <= 0.001)

    def split(self):
        min_impurity = float('inf')
        min_var_number = 0
        min_bound = 0
        min_left_y = None
        min_right_y = None
        for variable in self.variables.items():
            bounds = variable[1]
            var_number = variable[0]
            for bound in bounds:
                # print var_number, bound
                y_left = self.y[self.X.transpose()[var_number] <= bound]
                y_right = self.y[self.X.transpose()[var_number] > bound]
                if len(y_left) == 0 or len(y_right) == 0:
                    continue
                impurity = self.impurity(y_left, y_right)
                if min_impurity > impurity:
                    min_var_number = var_number
                    min_bound = bound
                    min_left_y = y_left
                    min_right_y = y_right
                    min_impurity = impurity
        self.variable = min_var_number
        self.bound = min_bound
        X_left = self.X[self.X.transpose()[self.variable] <= self.bound]
        X_right = self.X[self.X.transpose()[self.variable] > self.bound]
        # print self.tree_level
        # print min_impurity
        self.left_child = Node(self.max_tree_depth, self.max_X_length,
                               self.criterion, self.tree_level + 1,
                               self.node_id + 1)
        size_of_left_subtree = self.left_child.fit(X_left, min_left_y)
        self.right_child = Node(self.max_tree_depth, self.max_X_length,
                                self.criterion, self.tree_level + 1,
                                self.node_id + size_of_left_subtree + 1)
        self.right_child.fit(X_right, min_right_y)

    def predict(self, X_test):
        if self.leaf_node:
            return self.answer
        elif self.root_node:
            y_pred = []
            for x in X_test:
                if x[self.variable] <= self.bound:
                    y_pred.append(self.left_child.predict(x))
                else:
                    y_pred.append(self.right_child.predict(x))
            return np.array(y_pred)
        else:  # X_test is only one sample
            if X_test[self.variable] <= self.bound:
                return self.left_child.predict(X_test)
            else:
                return self.right_child.predict(X_test)

    def description(self):
        root_header = ""
        root_footer = ""
        if self.root_node:
            root_header = "strict digraph G {\n"
            root_footer = "}"
        node_description = ""
        if self.leaf_node:
            format_string = '{node_id}[label="{answer}"];\n'
            node_description = format_string.format(node_id=self.node_id,
                                                    answer=self.answer)
            return root_header + node_description + root_footer
        else:
            formatStr = """{node_id}[label="x_{variable} <= {bound}"];
{node_id} -> {left}[color=green];\n{node_id} -> {right}[color=red];\n"""
            node_description = formatStr.format(node_id=self.node_id,
                                                variable=self.variable,
                                                bound=self.bound,
                                                left=self.left_child.node_id,
                                                right=self.right_child.node_id)
            return (root_header + node_description +
                    self.left_child.description() +
                    self.right_child.description() + root_footer)
