import numpy as np

entries = np.array([1, 7, 5])
weight = np.array([0.8, 0.1, 0])


def sum(e, w):
    return e.dot(w)


s = sum(entries, weight)


def stepFunction(sum):
    if (sum > 1):
        return 1
    else:
        return 0


r = stepFunction(s)
print(r)