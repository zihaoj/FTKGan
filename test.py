# Common imports
from TrackingToy.TrackHelper import getPhiCircle
from TrackingToy.TrackHelper import drawTrack
from TrackingToy.detectorGeo import detectorGeo
import numpy as np
import os
np.random.seed(42)

# To plot pretty figures
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

rDet         = np.array([0.5, 1.0, 1.5, 2, 2.5,3.0])
nModPerLayer = np.array([14,   28,  42, 56, 70, 84])

xDet = np.linspace(-3, 3, 5000)
yDet = np.sqrt(0.5**2-xDet**2)


detGeo = detectorGeo()
detGeo.initSimple(rDet,nModPerLayer)
detGeo.setBeamSpot(X = 0.4, Y =0)

## Make Tower
nSSperLayer=[30,30,30,30,30,30]
detGeo.makeTower(phiCenter=0*np.pi/12,  phiWidth=np.pi/2, nSS=nSSperLayer)

np.random.seed(42)

solutions_test  = []
phiRange = (-3*np.pi/12, 3*np.pi/12)

for trkI in range(10):
    ptInv = np.random.uniform(0.01,0.5,1)[0]
    phi   = np.random.uniform(phiRange[0],phiRange[1],1)[0]
    q     = 2*np.random.randint(0,2,1)[0]-1
    pt = 1./ptInv
    
    xHits, yHits, modulesHit, actualHits = detGeo.getHits(pt,phi,q)
    solutions_test.append([xHits,yHits,pt,phi,q,modulesHit,actualHits])
                            



