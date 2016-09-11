import numpy as np
# from knn import MatrixBasedKNN
from window import Window


def accuracy(y_true, y_predict):
    score = 0.
    diff = np.array(y_true) - np.array(y_predict)
    for item in diff:
        if item == 0:
            score += 1
    score /= len(y_true)
    return score


def cross_validation(X, y, metric, k, kernel='optimal', cv_fold=5.):
    scores = []
    # performing random permutation on data
    perm = np.random.permutation(X.shape[0])
    X = X[perm]
    y = y[perm]
    # dividing into chunks
    chunks = []
    y_chunks = []
    start = 0
    chunk_size = X.shape[0]/cv_fold
    end = chunk_size
    while end < X.shape[0]:
        chunks.append(X[start:end])
        y_chunks.append(y[start:end])
        start = end
        end += chunk_size
    if (start < X.shape[0]):
        chunks.append(X[start:])
        y_chunks.append(y[start:])
    # calculating accuracy for each chunk
    for i in range(len(chunks)):
        # for knn cross-validation
        # knn = MatrixBasedKNN(num_loops=0)
        # knn = knn.fit(np.concatenate(chunks[:i]+chunks[i+1:],axis=0),
        #        np.concatenate(y_chunks[:i]+y_chunks[i+1:],axis=0), metric)
        # y_pred = knn.predict(chunks[i],k)

        # for window cross-validation
        window = Window()
        window = window.fit(chunks[i], np.concatenate(chunks[:i]+chunks[i+1:], axis=0),
                            np.concatenate(y_chunks[:i]+y_chunks[i+1:], axis=0), k, metric, kernel)
        y_pred = window.predict()
        scores.append(accuracy(y_chunks[i], y_pred))
    return np.mean(scores)
