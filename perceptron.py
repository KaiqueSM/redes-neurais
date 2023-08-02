entries = [1, 7, 5]
weight = [0.8, 0.1, 0]


def sum(e, w):
    s = 0
    for i in range(3):
        s += e[i] * w[i]
    return s


s = sum(entries, weight)


def stepFunction(sum):
    if (sum > 1):
        return 1
    else:
        return 0


r = stepFunction(s)
print(r)