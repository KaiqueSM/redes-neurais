from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets.supervised import SupervisedDataSet
from pybrain.supervised.trainers.backprop import BackpropTrainer
from pybrain.structure.modules.softmax import SoftmaxLayer
from pybrain.structure.modules.sigmoidlayer import SigmoidLayer

# network = buildNetwork(
#     2, 3, 1,
#     outclass=SoftmaxLayer,
#     hiddenclass=SigmoidLayer,
#     bias=False
# )
# print(network['in'])
# print(network['hidden0'])
# print(network['out'])
# print(network['bias'])

network = buildNetwork(2, 3, 1)
base = SupervisedDataSet(2, 1)
base.addSample((0, 0), (0, ))
base.addSample((0, 1), (1, ))
base.addSample((1, 0), (1, ))
base.addSample((1, 1), (0, ))

# print(base['input'])
# print(base['target'])

training = BackpropTrainer(
    network,
    dataset=base,
    learningrate=0.1,
    momentum=0.6
)

trainingTime = 30000

for i in range(1, trainingTime):
    error = training.train()
    if i % 1000 == 0:
        print("Error: ", error)

print(network.activate([0, 0]))
print(network.activate([0, 1]))
print(network.activate([1, 0]))
print(network.activate([1, 1]))
