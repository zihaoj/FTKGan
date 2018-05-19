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



def Effcal_pairwise(bankA, bankB):
    '''
    pairwise efficiency cal
    input:

    bankA: dic, first bank
    bankB: dic, second bank

    returns:
    an efficiency in float
    '''

    patA = sorted(bankA["bank"].keys() )
    patB = sorted(bankB["bank"].keys() )

    match = 0.0

    ## keep two pointers and move forward
    i = 0
    j = 0
    while (i < len(patA) and j < len(patB) ):

        if patA[i] == patB[j]:
            match += 1
            i += 1
            j += 1
            
        elif patA[i] < patB[j] and i != len(patA):
            i += 1

        elif patA[i] > patB[j] and j != len(patB):
            j += 1

    print match/len(patB)

    return match/len(patB)
    

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
        Effcal_pairwise(baseline, tests[t])


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

    bank_baseline = "PatternBanks/BankbeamX0_beamY0_Size10000_Phi-0.79-0.79_train.pickle"
    bank_tests    =["PatternBanks/BankbeamX0_beamY0_Size10000_Phi-0.79-0.79_test.pickle",
                    "PatternBanks/BankbeamX0.1_beamY0_Size10000_Phi-0.79-0.79_test.pickle",
                    "PatternBanks/BankbeamX0.1_beamY0.1_Size10000_Phi-0.79-0.79_test.pickle"]
    
    #Effcal_wrt_base(bank_baseline, bank_tests)
    Effcal_XYgrid(bank_baseline)    

__main__()    
