import tensorflow as tf
import math

class config():
  

    # model name
    modelname = "test"
    
    
    # model and training config
    batch_size  = 32
    lr_disc     = 1e-3
    lr_gen      = 1e-3
    lamb_lip    = 0.5

    seq_len   = 6
    embedsize = 2
    latent_dim = 2

    iterations     = 100000
    mon_frequency  = 10
    ngen_neurons   = 16
    ndisc_neurons  = 16
    use_aux_info   = True
    aux_info       = ["charge", "ptInv", "phi"]
    gen_type       = "GRU"

    
    # bank parameters
    truth_patterns = ["PatternBanks/BankbeamX0_beamY0_Size100000_Phi-0.79-0.79_train.pickle"]
    phiRange       = [ -1*math.pi/4, 1*math.pi/4 ]
    

    output_bank_size    = int(10**6)

    
    # output config
    output_path  = "GanResults/" + modelname +"/"
