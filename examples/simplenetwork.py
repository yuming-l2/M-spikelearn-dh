from spikelearn import SpikingNet, SpikingLayer, BaseSynapse
import numpy as np

snn = SpikingNet()
sl = SpikingLayer(10, 4)
syn = BaseSynapse(10, 10, np.random.random((10,10)))

snn.add_input("input1")
snn.add_layer(sl, "l1")
snn.add_synapse("l1", syn, "input1")
snn.add_output("l1")

u = 2*np.random.random(10)
for i in range(10):
    s = snn(2*np.random.random(10))
    print(s)

