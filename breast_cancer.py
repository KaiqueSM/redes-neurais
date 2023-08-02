import numpy as np
from sklearn import datasets

def sigmoid(num):
    return 1 / (1 + np.exp(-num))


def sigmoidDerivative(sig):
    return sig * (1 - sig)


base = datasets.load_breast_cancer()


entries = base.data
outingsValues = base.target
# mapeando valores para o array em forma de matriz
outings = np.empty([599, 1], dtype=int)
for i in range(599):
    outings[i] = outingsValues[i]

# pesos aleatorios
weights0 = 2 * np.random.random((30, 5)) - 1
weights1 = 2 * np.random.random((5, 1)) - 1

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
