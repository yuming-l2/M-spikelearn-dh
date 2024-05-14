from spikelearn import SpikingLayer
from spikelearn.generators import Poisson

import numpy as np
import matplotlib.pyplot as pt


sl = SpikingLayer(10, 4)

vlist = []
spikelist = []

spikegen = Poisson(10, np.random.random(10))

for _ in range(50):

    u = spikegen()
    spikelist.append(sl(2*u))
    vlist.append(sl.v)
    print(sl.s)

varr = np.stack(vlist, axis=0)

for i in range(10):
    pt.plot(varr[:,i]+1.1*i, "-k")
pt.xlabel("# steps")
pt.ylabel("Membrane voltage")
pt.show()

