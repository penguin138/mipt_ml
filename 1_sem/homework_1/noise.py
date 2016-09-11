from sklearn.neighbors import KDTree


def margin(indices, k, X, y):
    margins = []
    kd_tree = KDTree(X)
    for img_index in indices:
        margin = 0
        in_class = 0
        # most_frequent_class = 0
        current_class = y[img_index]
        # print current_class
        dists, neighbour_indices = kd_tree.query(X[img_index].reshape((1, X[img_index].shape[0])),
                                                 k)
        for index in neighbour_indices[0]:
            # print y[index]
            if y[index] == current_class:
                in_class += 1
        neighbour_dict = {}
        for index in neighbour_indices[0]:
            if y[index] in neighbour_dict:
                neighbour_dict[y[index]] += 1
            else:
                neighbour_dict[y[index]] = 1
        neighbour_dict.pop(current_class)
        if neighbour_dict:
            most_frequent = max(neighbour_dict.items(), key=lambda x: x[1])[1]
        margin = in_class - most_frequent
        margins.append(margin)
    return margins


def margin_new(indices, k, X, y):
    margins = []
    kd_tree = KDTree(X)
    for img_index in indices:
        margin = 0
        dist_to_class = 0
        dist_to_others = 0
        current_class = y[img_index]
        dists, neighbour_indices = kd_tree.query(X[img_index].reshape((1, X[img_index].shape[0])),
                                                 k)
        classes = {}
        for i in xrange(neighbour_indices[0].shape[0]):
            index = neighbour_indices[0][i]
            if y[index] in classes:
                classes[y[index]] += dists[0][i]
            else:
                classes[y[index]] = dists[0][i]
        dist_to_class = classes[current_class]
        classes.pop(current_class)
        # print classes.items()
        if classes:
            dist_to_others = min(classes.items(), key=lambda x: x[1])[1]
        margin = dist_to_class - dist_to_others
        margins.append(margin)
    return margins
