import numpy as np


def sigmoid(num):
    return 1 / (1 + np.exp(-num))


def sigmoidDerivative(sig):
    return sig * (1 - sig)


entries = np.array([
    [0, 0], [0, 1], [1, 0], [1, 1]
])
outings = np.array([
    [0], [1], [1], [0]
])
# weights0 = np.array([
#     [-0.424, -0.740, -0.961],
#     [0.358, -0.577, -0.469]
# ])
# weights1 = np.array([
#     [-0.017], [-0.893], [0.148]
# ])

# pesos aleatorios
weights0 = 2 * np.random.random((2, 3)) - 1
weights1 = 2 * np.random.random((3, 1)) - 1

learningFee = 0.3
momentum = 1

trainingTime = 100000
outLayer = ''

for time in range(trainingTime):
    entryLayer = entries

    # calculando camada oculta
    synapses0 = np.dot(entryLayer, weights0)
    hiddenLayer = sigmoid(synapses0)

    # calculando saída
    synapses1 = np.dot(hiddenLayer, weights1)
    outLayer = sigmoid(synapses1)

    # calculando erro
    errorOutLayer = outings - outLayer
    errorAverage = np.mean(np.abs(errorOutLayer))
    print("Erro: " + str(errorAverage))

    # calculando delta de saídas
    sigmoidDerivatives = sigmoidDerivative(outLayer)
    outDeltas = errorOutLayer * sigmoidDerivatives

    # calculo de delta da camada oculta
    weights1Transposed = weights1.T
    deltaOutXWeight = outDeltas.dot(weights1Transposed)
    hiddenLayerDelta = deltaOutXWeight * sigmoidDerivative(hiddenLayer)

    # calculo de novos pesos de camada oculta - saida
    hiddenLayerTransposed = hiddenLayer.T
    newWeights1 = hiddenLayerTransposed.dot(outDeltas)
    weights1 = (weights1 * momentum) + (learningFee * newWeights1)

    # calculo de delta da camada de entrada
    entryLayerTransposed = entryLayer.T
    newWeights0 = entryLayerTransposed.dot(hiddenLayerDelta)
    weights0 = (weights0 * momentum) + (newWeights0 * learningFee)

    # hiddenDelta = sigmoidDerivatives * weights1 * outDeltas
    # print(hiddenDelta)
    # print(errorAverage)


print(weights0)
print(outLayer)
