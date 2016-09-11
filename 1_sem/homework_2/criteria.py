from math import log


def gini(Y):
    p0 = len(Y[Y == 0])
    p1 = len(Y[Y == 1])
    length = float(len(Y))
    if length > 0:
        p0 /= length
        p1 /= length
        return p1 * (1 - p1) + p0 * (1 - p0)
    else:
        return 0


def twoing(Y_left, Y_right):
    len_left = float(len(Y_left))
    len_right = float(len(Y_right))
    l_0 = len(Y_left[Y_left == 0])
    l_1 = len(Y_left[Y_left == 1])
    r_0 = len(Y_right[Y_right == 0])
    r_1 = len(Y_right[Y_right == 1])
    diff_sum = (abs(l_0/len_left - r_0/len_right) +
                abs(l_1/len_left - r_1/len_right))
    value = (len_left*len_right/(len_left+len_right)**2) * diff_sum**2
    # print value
    return value


def entropy(Y):
    p0 = float(len(Y[Y == 0]))
    p1 = float(len(Y[Y == 1]))
    # length = float(len(Y))
    coef_0, coef_1 = 0, 0
    if p0 != 0:
        coef_0 = p0 * log(p0)
    if p1 != 0:
        coef_1 = p1 * log(p1)
    return -(coef_1 + coef_0)
