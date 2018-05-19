########################################################################################
###  Zihao's code for pattern bank generation of different det config using "fullsim"
###  Detector demonstrations see: DetWithModules.ipynb
########################################################################################

import time
import pickle
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
import os

from TrackingToy.TrackHelper import getPhiCircle
from TrackingToy.TrackHelper import drawTrack
from TrackingToy.detectorGeo import detectorGeo
from TrackingToy.TrackHelper import patternID



def PattGen( ntrk, beamX, beamY, rDet, nModPerLayer, phiRange, nSSperLayer, banktype, random=True ):

    if random:
        np.random.seed( int(time.time()) )
    else:
        np.random.seed( 10 )
    
    '''
    pattern bank generation for a detector specification 

    input:
    ntrk:  integer, number of randomly generated tracks to be used for pattern bank generation
    beamX: float,   beam spot position X
    beamY: float,   beam spot position Y
    rDet:  list[float], radius of each layer of tracker
    nModPerLayer:  list[int], number of modules per layer
    phiRange: list[float], phi range of the generated tower
    nSSperLayer: list[int], number of super strips per layer
    banktype: str,  useage of the bank: train or test?

    '''

    detGeo = detectorGeo()
    detGeo.initSimple(rDet,nModPerLayer)
    detGeo.setBeamSpot(X = beamX, Y =beamY)
    detGeo.makeTower(phiCenter=0*np.pi/12,  phiWidth=np.pi/2, nSS=nSSperLayer)


    ntrk_divide_5 = int(ntrk/5)
    print ("Making Patterns: {!s} trks, beamX={!s}, beamY={!s}" ).format(ntrk, beamX, beamY)

    solutions_pattGen = []
    for trkI in range(ntrk):

        if trkI % ntrk_divide_5 == 0:
            print "Simulating tracks {!s}% ".format(trkI/ntrk_divide_5*20)
                    
        ptInv = np.random.uniform(0.01,0.5,1)[0]
        phi   = np.random.uniform(phiRange[0],phiRange[1],1)[0]
        q     = 2*np.random.randint(0,2,1)[0]-1
        pt = 1./ptInv

        xHits, yHits, modulesHit, actualHits = detGeo.getHits(pt,phi,q)
        solutions_pattGen.append([xHits,yHits,pt,phi,q,modulesHit,actualHits])


    # The pattern bank
    patternBank = {}
    patternBankTrigSeq = {}

    # For plotting
    passedTracks = 0
    passedTracksSamples = np.array([],float)
    patternBankSize = np.array([],float)
    
    
    for sItr in range(ntrk):

        if trkI % ntrk_divide_5 == 0:
            print "Collecting patterns {!s}% ".format(sItr/ntrk_divide_5*20)
        
        s = solutions_pattGen[sItr]

        ### for towers we would only consider tracks which yield hits on all layers
        if not detGeo.hitsAllInTower(s[6]): continue
        hitList = []
        realHits = s[6]
        for rHit in realHits:
            hitList.append(  [1,rHit[0],rHit[1],rHit[2], rHit[3]] )

        passedTracks += 1
        hitsWithSSIDs = detGeo.addSSIDs(hitList)
        thisSSIDs = []
        thisSSIDs_trig = [ [] for i in range(len(nSSperLayer)) ]
        
        for _ in range(len(rDet)): thisSSIDs.append([])
            
        for hinfo in hitsWithSSIDs:
            layer = int(hinfo[1])
            thisSSIDs[layer].append(hinfo[5])

            center = 1./float(nSSperLayer[layer])*(phiRange[1]-phiRange[0])/2
            rad = hinfo[5]/float(nSSperLayer[layer])*(phiRange[1]-phiRange[0]) +center
            thisSSIDs_trig[layer] = [math.cos(rad), math.sin(rad) ]
            
        goodTrack = True
        for layItr, layerSSIDs in enumerate(thisSSIDs):
            if len(layerSSIDs) == 0:
                goodTrack = False
                    
            elif len(layerSSIDs) == 1:
                pass # all good
            else:
                while(not len(thisSSIDs[layItr]) == 1):
                    thisSSIDs[layItr].pop(-1)
        if not goodTrack: continue
        thisPatternID = patternID(thisSSIDs,nSSperLayer)

        
        if thisPatternID not in patternBank:
            patternBank[thisPatternID] = 0
            ## we also create a map from real number thisPatternID -> trig transformed sequence ssid -> [cos(ssid/nssid*phirange), sin(ssid/nssid*phirange)  ]
            patternBankTrigSeq [ thisPatternID ] = np.array(thisSSIDs_trig)

        patternBank[thisPatternID] += 1
        
        
        # Statistics
        if passedTracks %10 == 0:
            passedTracksSamples = np.append(passedTracksSamples,passedTracks)
            patternBankSize     = np.append(patternBankSize, len(patternBank))


    bankname = "beamX{!s}_beamY{!s}_Size{!s}_Phi{!s}-{!s}_{!s}".format(beamX, beamY, ntrk, round(phiRange[0],2), round(phiRange[1],2), banktype)

    ### saving the pattern bank
    patternsBankToLoad = {}
    patternsBankToLoad["rDet"]         = rDet
    patternsBankToLoad["nSSperLayer"]  = nSSperLayer
    patternsBankToLoad["beamX"]        = beamX
    patternsBankToLoad["beamY"]        = beamY
    patternsBankToLoad["phi_low"]      = phiRange[0]
    patternsBankToLoad["phi_high"]     = phiRange[1]    
    patternsBankToLoad["bank"]         = patternBank
    patternsBankToLoad["bankTrigSeq"]  = patternBankTrigSeq
    patternsBankToLoad["passedTracks"] = passedTracksSamples
    patternsBankToLoad["bankSize"]     = patternBankSize

    with open('PatternBanks/Bank'+bankname+'.pickle', 'wb') as handle:
        pickle.dump(patternsBankToLoad, handle, protocol=pickle.HIGHEST_PROTOCOL)

    ### sanity check plots
    # npat vs ntrk plot
    plt.plot(passedTracksSamples,patternBankSize,"k")
    plt.xlabel("Tracks")
    plt.ylabel("nPatterns")
    plt.savefig( "FigBanks/NPat_Vs_NTrk_" + bankname+ ".pdf" )
    plt.close()

    # pick 10 tracks to visualize
    solutions_pattGen
    fig, ax = plt.subplots(1, figsize=(10,10))
    ax.set_xlim((0, 4))
    ax.set_ylim((-2, 2))

    detGeo.drawTower(ax,detailed=True)
    for sItr in range(10):
        s = solutions_pattGen[sItr]

        if not detGeo.hitsAllInTower(s[6]): continue

        color = "b" if (s[4] < 0) else "r"
        plt.plot(s[0],s[1],color)
        plt.plot( [detGeo.beamX, s[0][0]], [detGeo.beamY, s[1][0]], color   )
        
        hitList = []
        realHits = s[6]
        for rHit in realHits:
            # hitITr / Layer / phiID / hit-X / hit-Y
            hitList.append(  [1,rHit[0],rHit[1],rHit[2], rHit[3]] )


        hitsWithSSIDs = detGeo.addSSIDs(hitList)
        
        thisSSIDs = np.array(hitsWithSSIDs)
        thisSSIDs = np.delete(hitsWithSSIDs,0,axis=1) # rm hitID
        thisSSIDs = np.delete(thisSSIDs,1,axis=1)     # rm phiID
        thisSSIDs = np.delete(thisSSIDs,1,axis=1)     # rm hit X
        thisSSIDs = np.delete(thisSSIDs,1,axis=1)     # rm hit Y

        plt.plot(s[0],s[1],'ko')
        detGeo.drawSSIDs(ax,thisSSIDs)

        for ih, rH in enumerate(realHits):
            plt.plot(rH[2],rH[3],'bo')
    plt.savefig( "FigBanks/Visualization_" + bankname+ ".pdf" )
    plt.close()

    
def __main__():

    rDet         = np.array([0.5, 1.0, 1.5, 2, 2.5,3.0])
    nModPerLayer = np.array([14,   28,  42, 56, 70, 84])
    phiRange     = (-3*np.pi/12, 3*np.pi/12)
    nSSperLayer  = [30,30,30,30,30,30]
    ntrk         = int(1e4)

    beamX = [0, 0.005, 0.01, 0.015, 0.02]
    beamY = [0, 0.005, 0.01, 0.015, 0.02]

    PattGen( ntrk=ntrk, beamX=0, beamY=0, rDet=rDet, nModPerLayer=nModPerLayer, phiRange=phiRange, nSSperLayer=nSSperLayer, banktype = "train" )
    
    '''
    for x in beamX:
        for y in beamY:
            #PattGen( ntrk=ntrk, beamX=0, beamY=0, rDet=rDet, nModPerLayer=nModPerLayer, phiRange=phiRange, nSSperLayer=nSSperLayer, banktype = "train" )
            PattGen( ntrk=ntrk, beamX=x, beamY=y, rDet=rDet, nModPerLayer=nModPerLayer, phiRange=phiRange, nSSperLayer=nSSperLayer, banktype = "test" )    
    '''

    
        
__main__()
    
