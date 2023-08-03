from pybrain.structure.networks import FeedForwardNetwork
from pybrain.structure.connections.full import FullConnection
from pybrain.structure.modules.linearlayer import LinearLayer
from pybrain.structure.modules.sigmoidlayer import SigmoidLayer
from pybrain.structure.modules.biasunit import BiasUnit


network = FeedForwardNetwork()

entryLayer = LinearLayer(2)
hiddenLayer = SigmoidLayer(3)
outLayer = SigmoidLayer(1)

bias1 = BiasUnit()
bias2 = BiasUnit()

network.addModule(entryLayer)
network.addModule(hiddenLayer)
network.addModule(outLayer)
network.addModule(bias1)
network.addModule(bias2)

hiddenEntry = FullConnection(entryLayer, hiddenLayer)
hiddenOut = FullConnection(hiddenLayer, outLayer)
hiddenBias = FullConnection(bias1, hiddenLayer)
biasOut = FullConnection(bias2, outLayer)

network.sortModules()

print(network)
print(hiddenEntry.params)
print(hiddenOut.params)
print(hiddenBias.params)
print(biasOut.params)
