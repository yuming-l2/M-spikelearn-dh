from spikelearn import SpikingLayer

import numpy as np
import matplotlib.pyplot as pt


sl = SpikingLayer(10, 4)

vlist = []
spikelist = []

for _ in range(50):

    u = 2*np.arange(0.1,1.1,0.1)
    spikelist.append(sl(u))
    vlist.append(sl.v)
varr = np.stack(vlist, axis=0)

for i in range(10):
    pt.plot(varr[:,i]+1.1*i, "-k")
pt.xlabel("# steps")
pt.ylabel("Membrane voltage")
pt.show()

