import numpy as np

entries = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
outings = np.array([0, 1, 1, 1])
weights = np.array([0.0, 0.0])
learningFee = 0.1


def stepFunction(out):
    if out >= 1:
        return 1
    else:
        return 0


def calcOut(registry):
    s = registry.dot(weights)
    return stepFunction(s)


# error = correctOut - calculateOut
# weight(n+1) = weight(n) + (learningFee * entry * error)
def learn():
    totalErrors = 1
    while totalErrors != 0:
        totalErrors = 0
        for i in range(len(outings)):
            out = calcOut(np.array(entries[i]))
            error = abs(outings[i] - out)
            totalErrors += error
            if error > 0:
                for w in range(len(weights)):
                    # print(str(weights[w])+" + ("+str(learningFee)+" * "+str(entries[i][w])+" * "+str(error)+")")
                    weights[w] = round(
                        (weights[w] + (learningFee * entries[i][w] * error)),
                        1
                    )
                    print("Peso atualizado: " + str(weights[w]))
        print("Total erros: " + str(totalErrors))


learn()

print("Neural network trained!")
print(calcOut(entries[0]))
print(calcOut(entries[1]))
print(calcOut(entries[2]))
print(calcOut(entries[3]))
