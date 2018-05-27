########################################################################################
###  Zihao's code for pattern bank efficiency evaluation
########################################################################################

import time
import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
from copy import deepcopy
from collections import OrderedDict
from operator import itemgetter



def Effcal_pairwise(bankA, bankB):
    '''
    pairwise efficiency cal
    input:

    bankA: dic, target bank
    bankB: dic, baseline bank

    returns:
    an efficiency in float
    '''

    ## First make sure we get the same number of patterns for bankB and bankA
    nBankA = len(bankA["bank"])    
    nBankB = len(bankB["bank"])

    patB = sorted(bankB["bank"].keys() )
    patA = []

    if nBankA>nBankB:
        d = OrderedDict(sorted(bankA["bank"].items(), key=itemgetter(1), reverse=True))
        for ipat in range(nBankB):
            patA.append( d.keys()[ipat]  )
        patA = sorted(patA)
                            
    else:
        patA = sorted(bankA["bank"].keys() )


    matched_patterns = 0.0
    matched_trks = 0.0

    All_InvpT     = np.array([ v for tup in bankB["ptInv"].values() for v in tup])
    All_q         = np.array([ v for tup in bankB["q"].values() for v in tup])
    All_phi       = np.array([ v for tup in bankB["phi"].values() for v in tup])    

    Matched_InvpT = []
    Matched_q     = []
    Matched_phi   = []    

    ## keep two pointers and move forward
    i = 0
    j = 0
    while (i < len(patA) and j < len(patB) ):

        if int(patA[i]) == int(patB[j]):

            for invpt in bankB["ptInv"][patB[j]]:
                Matched_InvpT.append( invpt   )
            for q in bankB["q"][patB[j]]:
                Matched_q.append(  q )
            for phi in bankB["phi"][patB[j]]:
                Matched_phi.append( phi )
            
            matched_patterns += 1
            matched_trks     += bankB["bank"][ patB[j] ]
            i += 1
            j += 1

            
        elif patA[i] < patB[j] and i != len(patA):
            i += 1

        elif patA[i] > patB[j] and j != len(patB):
            j += 1

    plt.hist(All_phi, 50,  normed = 0, edgecolor='red', fill=False)
    plt.hist(Matched_phi, 50,  normed = 0, edgecolor='blue', fill = False)
    plt.show()
            
    plt.hist(All_InvpT, 50,  normed = 0, edgecolor='red', fill=False)
    plt.hist(Matched_InvpT, 50,  normed = 0, edgecolor='blue', fill = False)
    plt.show()

    plt.hist(All_q, 50,  normed = 0, edgecolor='red', fill=False)
    plt.hist(Matched_q, 50,  normed = 0, edgecolor='blue', fill = False)
    plt.show()
    
            
    print matched_patterns/len(patB)
    print matched_trks/ sum(bankB["bank"].values())

    return matched_patterns/len(patB)
    

def Effcal_wrt_base( bank_baseline, bank_tests ):

    '''
    pattern bank efficiency cal w.r.t. baseline
    input:

    bank_baseline:  str,  name of baseline bank
    bank_tests:  list[str], name of test banks

    returns:
    list of efficiency of test banks compared to baseline
    '''

    ## Loading baseline
    baseline =None
    with open(bank_baseline, 'rb') as handle:
        baseline = pickle.load(handle)

    ## Loading tests
    tests = {}
    for t in bank_tests:
        tmp = None
        with open(t, 'rb') as handle:
            tmp = pickle.load(handle)

        tests[t]= deepcopy(tmp)


    for t in bank_tests:
        Effcal_pairwise(tests[t], baseline)


def Effcal_XYgrid(bank_baseline):

    baseline =None
    with open(bank_baseline, 'rb') as handle:
        baseline = pickle.load(handle)

    beamX = [0, 0.005, 0.01, 0.015, 0.02]
    beamY = [0, 0.005, 0.01, 0.015, 0.02]

    banks = {}
    eff_dict   = {}
    eff_array  = np.zeros( [ len(beamX), len(beamY) ] )

    for x in beamX:
        for y in beamY:
            filename = "PatternBanks/BankbeamX{!s}_beamY{!s}_Size10000_Phi-0.79-0.79_test.pickle".format(str(x), str(y))

            tmp = None
            with open(filename, 'rb') as handle:
                tmp = pickle.load(handle)

            banks[ (x, y) ] = deepcopy(tmp)

    baseline_eff = Effcal_pairwise(banks[ (0,0) ], baseline)
    
    for ix, x in enumerate(beamX):
        for iy, y in enumerate(beamY):
            eff_dict[ (x, y) ] = Effcal_pairwise(banks[ (x,y) ], baseline) / baseline_eff
            eff_array[ -iy-1 ][ ix ] = eff_dict[ (x, y) ]

    fig, ax = plt.subplots()
    
    ax.set_xticks(np.arange(len(beamX)))
    ax.set_yticks(np.arange(len(beamY)))
    ax.set_xticklabels(beamX)
    ax.set_yticklabels(beamY[::-1])
    plt.xlabel("beamSpot Shift X (dm)")
    plt.ylabel("beamSpot Shift Y (dm)")
    
    im = ax.imshow(eff_array, cmap="YlGn")
    fig.colorbar(im)
    plt.title("Bank Efficiency vs. Beam Spot Movement")

    plt.show()
        

def __main__():

    bank_tests    =["PatternBanks/BankbeamX0_beamY0_Size10000_Phi-0.79-0.79_test.pickle",
                    "PatternBanks/BankbeamX0.1_beamY0_Size10000_Phi-0.79-0.79_test.pickle",
                    "PatternBanks/BankbeamX0.1_beamY0.1_Size10000_Phi-0.79-0.79_test.pickle"]

    bank_baseline = "PatternBanks/BankbeamX0_beamY0_Size10000_Phi-0.79-0.79_train.pickle"
    bank_tests    =["PatternBanks/BankbeamX0_beamY0_Size10000_Phi-0.79-0.79_test.pickle"]    
    gan_tests     = ["PatternBanks/BankGan.pickle"]

    Effcal_wrt_base(bank_baseline, gan_tests)
    #Effcal_wrt_base(bank_baseline, bank_tests)
    #Effcal_XYgrid(bank_baseline)    

__main__()    
