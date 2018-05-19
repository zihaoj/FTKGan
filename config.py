import tensorflow as tf
import math

class config():
  

    # model name
    modelname = "test"
    
    
    # model and training config
    batch_size  = 100 # number of steps used to compute each policy update
    lr          = 1e-3

    
    seq_len   = 6
    embedsize = 2
    latent_dim = 1

    iterations  = 100
    

    # bank parameters
    truth_patterns = ["PatternBanks/BankbeamX0_beamY0_Size10000_Phi-0.79-0.79_train.pickle"]
    phiRange       = [ -2*math.pi/4, 2*math.pi/4 ]

    
    # output config
    output_path  = "GanResults/" + modelname +"/"
