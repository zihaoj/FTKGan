import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell
from tensorflow.contrib.layers import fully_connected
import numpy as np
import pickle
from config import config
from TrackingToy.TrackHelper import patternID


## input sequence of [shape batch * length of sequence * embedding]

def build_Discriminator_GRU(inputs, aux_info, scope, num_neurons=16, reuse = False):
  
  with tf.variable_scope(scope, reuse = reuse):

    cell = GRUCell(num_neurons)
  
    outputs, states = tf.contrib.rnn.static_rnn(
      cell,
      tf.unstack( tf.transpose(inputs, [1,0,2]) ),
      dtype=tf.float32,

    )

    #last = outputs[-1]
    last   = outputs[-1]

    ## concact to get sequence
    outputs = tf.concat( outputs, axis=1)

    
    ## get discriminator score
    prediction = tf.contrib.layers.fully_connected(inputs =outputs, num_outputs = 1,
                                                   activation_fn = tf.sigmoid,
                                                   weights_initializer=tf.random_uniform_initializer(minval=-1, maxval=1),
                                                   biases_initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1) )

        
    aux = {}
    charge     = tf.contrib.layers.fully_connected(inputs = outputs, num_outputs = 1,
                                                   activation_fn = tf.nn.tanh,
                                                   weights_initializer=tf.random_uniform_initializer(minval=-1, maxval=1),
                                                   biases_initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1) )

    ptInv      = tf.contrib.layers.fully_connected(inputs = outputs, num_outputs = 1,
                                                   activation_fn = None,
                                                   weights_initializer=tf.random_uniform_initializer(minval=-1, maxval=1),
                                                   biases_initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1) )


    phi        = tf.contrib.layers.fully_connected(inputs = outputs, num_outputs = 1,
                                                   activation_fn = None,
                                                   weights_initializer=tf.random_uniform_initializer(minval=-1, maxval=1),
                                                   biases_initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1) )
    
    ## get charge aux task score
    ## two categories +1 and -1

    if "charge" in aux_info:
      aux["charge"]   = charge

      
    ## get ptinv aux task score
    ## continuous variable

    if "ptInv" in aux_info:
      aux["ptInv"]    = ptInv


    ## get phi aux task score
    ## continuous variable

    if "phi" in aux_info:
      aux["phi"]    = phi

      
    return prediction, aux


def build_Generator_GRU(inputs, aux_inputs, scope, phiRange, num_neurons=16, reuse = False):


  with tf.variable_scope(scope, reuse = reuse):
    gen_seq_length = inputs.get_shape()[1]
    
    cell = GRUCell(num_neurons)

    outputs = []
    output_rad = []
    output_seq = []
    

    #concact_inputs = tf.expand_dims( aux_inputs, axis = 1) ## add sequence dimension
    concact_inputs = tf.concat( [inputs, aux_inputs] , axis = 2 )
    

    outputs, states = tf.contrib.rnn.static_rnn(
      cell,
      tf.unstack( tf.transpose(concact_inputs, [1,0,2]) ),
      dtype=tf.float32,
    )
    
    for iq in xrange(gen_seq_length ):
      logits = tf.contrib.layers.fully_connected(inputs =outputs[iq], num_outputs = 1, activation_fn = tf.sigmoid,                                                 
                                                 weights_initializer=tf.random_uniform_initializer(minval=-1, maxval=1),
                                                 biases_initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1) )


      rad   = phiRange[0] + logits *( phiRange[1]-phiRange[0])

      rad_tmp = tf.expand_dims(rad, 1 )
      output_rad.append(rad_tmp)
      
      out_cos = tf.cos( rad)
      out_sin = tf.sin( rad)
      
      seq_tmp = tf.concat([ out_cos, out_sin], axis=1)
      seq_tmp = tf.expand_dims(seq_tmp, 1)
      output_seq.append(seq_tmp)


      
    return tf.concat( output_rad , axis = 1), tf.concat( output_seq , axis = 1)
      

def build_Generator_Dense(inputs, scope, phiRange, num_neurons=32, reuse = False):


  with tf.variable_scope(scope, reuse = reuse):

    gen_seq_length = inputs.get_shape()[1]
    
    outputs = []
    output_rad = []
    output_seq = []
    for iq in xrange(gen_seq_length):
      
      if iq ==0:
        outputs.append(tf.contrib.layers.fully_connected(inputs =inputs[:, iq, :], num_outputs = num_neurons, activation_fn = tf.sigmoid,                                   
                                                         weights_initializer=tf.random_uniform_initializer(minval=-1, maxval=1),
                                                         biases_initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1) ) )

      else:
        outputs.append(tf.contrib.layers.fully_connected(inputs =tf.concat([ inputs[:, iq, :], outputs[-1]], axis=1), num_outputs = num_neurons, activation_fn = tf.sigmoid,
                                                         weights_initializer=tf.random_uniform_initializer(minval=-1, maxval=1),
                                                         biases_initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1) ) )
        
    for iq in xrange(gen_seq_length ):
      logits = tf.contrib.layers.fully_connected(inputs =outputs[iq], num_outputs = 1, activation_fn = tf.sigmoid,                                                 
                                                 weights_initializer=tf.random_uniform_initializer(minval=-1, maxval=1),
                                                 biases_initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1) )


      rad   = phiRange[0] + logits *( phiRange[1]-phiRange[0])

      rad_tmp = tf.expand_dims(rad, 1 )
      output_rad.append(rad_tmp)
      
      out_cos = tf.cos( rad)
      out_sin = tf.sin( rad)
      
      seq_tmp = tf.concat([ out_cos, out_sin], axis=1)
      seq_tmp = tf.expand_dims(seq_tmp, 1)
      output_seq.append(seq_tmp)


      
    return tf.concat( output_rad , axis = 1), tf.concat( output_seq , axis = 1)



class FTKGan(object):
  """
  Abstract Class for FTKGan
  """
  def __init__(self, config):
    """
    """

    ## get configuration parameters
    self.config     = config
    self.seq_len    = self.config.seq_len
    self.batch_size = self.config.batch_size
    self.embedsize  = self.config.embedsize
    self.latent_dim = self.config.latent_dim
    self.lr_disc    = self.config.lr_disc
    self.lr_gen     = self.config.lr_gen
    self.iterations = self.config.iterations
    self.lamb_lip   = self.config.lamb_lip    

    self.ngen_neurons  = self.config.ngen_neurons
    self.ndisc_neurons = self.config.ndisc_neurons
    self.use_aux_info  = self.config.use_aux_info
    self.gen_type      = self.config.gen_type
    self.aux_info      = self.config.aux_info    
    self.aux_dim       = len(self.config.aux_info)


    self.truth_patterns  = self.config.truth_patterns
    self.phiRange        = self.config.phiRange
    self.phiRes          = []
    self.nSSperLayer     = None

    ## get network graphs initialized
    self.add_placeholders()
    self.build_generator_and_discriminator()
    self.add_loss_op()
    self.add_optimizer_op()


    ## Loading banks
    ## Concactenate all banks
    patternSequence = []
    chargeSequence = []
    ptInvSequence  = []
    phiSequence  = []    
    
    for t in self.truth_patterns:
      tmp = None
      with open(t, 'rb') as handle:
        tmp = pickle.load(handle)

      for p in tmp["bank"].keys():
        patternSequence.append( tmp["bankTrigSeq"][p]  )
        chargeSequence.append( [  sum(tmp["q"][p])/float( len(tmp["q"][p]) )]         *   self.seq_len )   # use average
        ptInvSequence.append ( [ (sum(tmp["ptInv"][p])/float( len(tmp["ptInv"][p])) -0.01) /  (0.5-0.01) ]  *self.seq_len )
        phiSequence.append   ( [ (sum(tmp["phi"][p])/float( len(tmp["phi"][p]))-self.phiRange[0]) / (self.phiRange[1]-self.phiRange[0]) ]      *self.seq_len )

      self.nSSperLayer = tmp["nSSperLayer"]

    
    for il in range(len(self.nSSperLayer)):
      self.phiRes.append( (self.phiRange[1]-self.phiRange[0])/ self.nSSperLayer[il] )

    self.real_patterns = np.array( patternSequence )[:, 0:self.seq_len,:]

    self.charge       = np.array( chargeSequence )
    self.charge       = np.expand_dims(self.charge, axis = 2)

    self.ptInv        = np.array( ptInvSequence)
    self.ptInv        = np.expand_dims(self.ptInv, axis = 2)

    self.phi          = np.array( phiSequence)
    self.phi          = np.expand_dims(self.phi, axis = 2)

    self.ninput        = self.real_patterns.shape[0]
    self.output_bank_size  = self.config.output_bank_size
    
    
  def add_placeholders(self):
    """
    Adds placeholders to the graph
    Set up the observation, action, and advantage placeholder
    """

    self.disc_real_input_placeholder       = tf.placeholder(tf.float32, shape=(self.batch_size, self.seq_len, self.embedsize),  name="real_seq_placeholder" )
    self.disc_aux_input_placeholder        = tf.placeholder(tf.float32, shape=(self.batch_size, self.seq_len, self.aux_dim),    name="real_aux_placeholder" )

    self.gen_latent_input_placeholder      = tf.placeholder(tf.float32, shape=(self.batch_size, self.seq_len, self.latent_dim), name="gen_latent_placeholder" )
    self.gen_aux_input_placeholder         = tf.placeholder(tf.float32, shape=(self.batch_size, self.seq_len, self.aux_dim),    name="gen_aux_placeholder" )

  
  def build_generator_and_discriminator(self):
    """
    """

    self.D = build_Discriminator_GRU
    self.G = None
    if self.gen_type == "GRU":
      self.G = build_Generator_GRU
    if self.gen_type == "Dense":
      self.G = build_Generator_Dense
    

  
  def add_loss_op(self):

    ## Get the generator
    self.generator_rad, self.generator = self.G( self.gen_latent_input_placeholder, self.gen_aux_input_placeholder,
                                                 "Generator",     self.phiRange,  num_neurons = self.ngen_neurons )

    self.disc_real, self.disc_real_aux = self.D( self.disc_real_input_placeholder, self.aux_info, 
                                                 "Discriminator", num_neurons = self.ndisc_neurons)

    self.disc_fake, self.disc_fake_aux = self.D( self.generator,                   self.aux_info,
                                                 "Discriminator", num_neurons = self.ndisc_neurons, reuse = True )

    ## Costs
    self.costs = {}

    ## WGAN lipschitz-penalty
    '''
    Could potentially use lipschitz penalty? 
    WGAN formulation https://arxiv.org/pdf/1701.04862.pdf

    self.costs["disc"] = -tf.reduce_mean( self.disc_real ) + tf.reduce_mean( self.disc_fake )
    self.costs["gen"]  = -tf.reduce_mean( self.disc_fake )

    alpha = tf.random_uniform(
      shape=[self.batch_size, 1, 1],
      minval=0.,
      maxval=1.
    )
    differences  = self.generator - self.disc_real_input_placeholder
    interpolates = self.disc_real_input_placeholder+ (alpha * differences)
    gradients = tf.gradients( self.D( interpolates, "Discriminator", num_neurons = self.ndisc_neurons, reuse = True ), [interpolates])[0]
                                  
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients)) )
    self.gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
    self.costs["disc"] = self.costs["disc"] + self.lamb_lip * self.gradient_penalty
    '''

    ## GAN primary task loss
    self.costs["disc"] = -tf.reduce_mean(tf.log( self.disc_real ) + tf.log(1. - self.disc_fake) )
    self.costs["gen"] =  -tf.reduce_mean(tf.log( self.disc_fake) )

    ## GAN aux task loss
    self.costs["disc_aux"] = 0
    self.costs["gen_aux"] = 0


    def poisson(y_true, y_pred):
      return tf.reduce_mean(y_pred - y_true * tf.log(y_pred + 1e-7))
    
    if "charge" in self.aux_info:
      
      self.costs["disc_real_aux_charge"] = tf.reduce_mean(  tf.losses.mean_squared_error( labels= self.disc_aux_input_placeholder[:,0,0],
                                                                                          predictions= self.disc_real_aux["charge"][:,0] ))
      self.costs["disc_fake_aux_charge"] = tf.reduce_mean(  tf.losses.mean_squared_error( labels= self.gen_aux_input_placeholder[:,0,0]  ,
                                                                                          predictions=self.disc_fake_aux["charge"][:,0] ))
        
      self.costs["disc_aux"] +=  (self.costs["disc_real_aux_charge"] + self.costs["disc_fake_aux_charge"])
      self.costs["gen_aux"]  +=  self.costs["disc_fake_aux_charge"]

    if "ptInv" in self.aux_info:
      self.costs["disc_real_aux_ptInv"] = tf.reduce_mean(  tf.losses.mean_squared_error( labels= self.disc_aux_input_placeholder[:,0,1] ,
                                                                                         predictions=self.disc_real_aux["ptInv"][:,0] ))
      self.costs["disc_fake_aux_ptInv"] = tf.reduce_mean(  tf.losses.mean_squared_error( labels= self.gen_aux_input_placeholder[:,0,1],
                                                                                         predictions=self.disc_fake_aux["ptInv"][:,0] ))
        
      self.costs["disc_aux"] +=  (self.costs["disc_real_aux_ptInv"] + self.costs["disc_fake_aux_ptInv"])
      self.costs["gen_aux"]  +=  self.costs["disc_fake_aux_ptInv"]


    if "phi" in self.aux_info:

      self.costs["disc_real_aux_phi"] = tf.reduce_mean(  tf.losses.mean_squared_error( labels= self.disc_aux_input_placeholder[:,0,2] ,
                                                                                       predictions=self.disc_real_aux["phi"][:,0] ))
      self.costs["disc_fake_aux_phi"] = tf.reduce_mean(  tf.losses.mean_squared_error( labels= self.gen_aux_input_placeholder[:,0,2],
                                                                                       predictions=self.disc_fake_aux["phi"][:,0] ))

      self.costs["disc_aux"] +=  (self.costs["disc_real_aux_phi"] + self.costs["disc_fake_aux_phi"])
      self.costs["gen_aux"]  +=  self.costs["disc_fake_aux_phi"]


      
    ## GAN combine losses
    if self.use_aux_info:
      self.costs["disc"] += self.costs["disc_aux"]
      self.costs["gen"]  += self.costs["gen_aux"]    
        
        
    ## Prepare cross check and monitoring quantities
    self.scores   = {}
    self.scores["disc"] = tf.reduce_mean( self.disc_real )    
    self.scores["gen"]  = tf.reduce_mean( self.disc_fake )

    self.grads    = {}
    self.grads["disc"]  = tf.gradients( self.disc_real,  [ self.disc_real_input_placeholder])
    self.grads["gen"]   = tf.gradients( self.disc_fake,  [ self.gen_latent_input_placeholder])

  
  def add_optimizer_op(self):

    gen_pars  = [p for p in tf.trainable_variables() if "Generator" in p.name]
    disc_pars = [p for p in tf.trainable_variables() if "Discriminator" in p.name]

    self.GRUdisc_weight = None
    self.FCdisc_weight  = None
    self.GRUgen_weight = None
    self.FCgen_weight  = None
    
    for p in disc_pars:
      if p.name == "Discriminator/rnn/gru_cell/gates/weights:0":
        self.GRUdisc_weight =  p
      if p.name == "Discriminator/fully_connected/weights:0":
        self.FCdisc_weight =  p

    for p in gen_pars:
      if p.name == "Generator/rnn/gru_cell/gates/weights:0":
        self.GRUgen_weight =  p
      if p.name == "Generator/fully_connected/weights:0":
        self.FCgen_weight =  p


    self.gen_train_op  = tf.train.AdamOptimizer(learning_rate=self.lr_gen).minimize(self.costs["gen"], var_list= gen_pars)
    self.disc_train_op = tf.train.AdamOptimizer(learning_rate=self.lr_disc).minimize(self.costs["disc"], var_list= disc_pars)


  def add_tb_summary(self):

    self.disc_score_placeholder = tf.placeholder(tf.float32, shape=(), name="discriminator_score")
    self.gen_score_placeholder  = tf.placeholder(tf.float32, shape=(), name="generator_score")    

    tf.summary.scalar("Discriminator/score", self.disc_score_placeholder)
    tf.summary.scalar("Generator/score",     self.gen_score_placeholder)
    
    self.disc_grad_placeholder = tf.placeholder(tf.float32, shape=(), name="discriminator_gradients")            
    self.gen_grad_placeholder  = tf.placeholder(tf.float32, shape=(), name="generator_gradients")

    tf.summary.scalar("Discriminator/grad",  self.disc_grad_placeholder)    
    tf.summary.scalar("Generator/grad",      self.gen_grad_placeholder)
    
    self.disc_cost_placeholder = tf.placeholder(tf.float32, shape=(), name="discriminator_cost")
    self.gen_cost_placeholder  = tf.placeholder(tf.float32, shape=(), name="generator_cost")    

    tf.summary.scalar("Discriminator/cost",  self.disc_cost_placeholder)
    tf.summary.scalar("Generator/cost",      self.gen_cost_placeholder)

    if "charge" in self.aux_info:

      self.disc_charge_cost_placeholder = tf.placeholder(tf.float32, shape=(), name="discriminator_charge_cost")
      self.gen_charge_cost_placeholder  = tf.placeholder(tf.float32, shape=(), name="generator_charge_cost")

      tf.summary.scalar("Discriminator/cost_charge",  self.disc_charge_cost_placeholder)
      tf.summary.scalar("Generator/cost_charge",      self.gen_charge_cost_placeholder)

    if "ptInv" in self.aux_info:

      self.disc_ptInv_cost_placeholder = tf.placeholder(tf.float32, shape=(), name="discriminator_ptInv_cost")
      self.gen_ptInv_cost_placeholder  = tf.placeholder(tf.float32, shape=(), name="generator_ptInv_cost")    
    
      tf.summary.scalar("Discriminator/cost_ptInv",  self.disc_ptInv_cost_placeholder)
      tf.summary.scalar("Generator/cost_ptInv",      self.gen_ptInv_cost_placeholder)


    if "phi" in self.aux_info:

      self.disc_phi_cost_placeholder = tf.placeholder(tf.float32, shape=(), name="discriminator_phi_cost")
      self.gen_phi_cost_placeholder  = tf.placeholder(tf.float32, shape=(), name="generator_phi_cost")    
    
      tf.summary.scalar("Discriminator/cost_phi",  self.disc_phi_cost_placeholder)
      tf.summary.scalar("Generator/cost_phi",      self.gen_phi_cost_placeholder)

      
    self.merged = tf.summary.merge_all()
    self.file_writer = tf.summary.FileWriter(self.config.output_path+"logs", self.sess.graph)


  def record_summary(self, it):

    """
    Add summary to tfboard                                                                                                                                                                                  
    """
    fd = {
      self.disc_score_placeholder: self.disc_score,
      self.gen_score_placeholder:  self.gen_score,

      self.disc_grad_placeholder:  self.disc_grads,      
      self.gen_grad_placeholder:   self.gen_grads,

      self.disc_cost_placeholder:  self.disc_cost,
      self.gen_cost_placeholder:   self.gen_cost,      

    }

    if "charge" in self.aux_info:
      fd [self.disc_charge_cost_placeholder] =   self.disc_charge_cost
      fd [self.gen_charge_cost_placeholder]  =   self.gen_charge_cost

    if "ptInv" in self.aux_info:
      fd [self.disc_ptInv_cost_placeholder]  =   self.disc_ptInv_cost
      fd [self.gen_ptInv_cost_placeholder]   =   self.gen_ptInv_cost

    if "phi" in self.aux_info:
      fd [self.disc_phi_cost_placeholder]  =   self.disc_phi_cost
      fd [self.gen_phi_cost_placeholder]   =   self.gen_phi_cost

    
    summary = self.sess.run(self.merged, feed_dict=fd)

    # tensorboard stuff
    self.file_writer.add_summary(summary, it)

    
  def initialize(self):
    """
    """

    # create tf session
    self.sess = tf.Session()

    # prepare tensor board monitoring
    self.add_tb_summary()

    ## start the session
    init = tf.global_variables_initializer()
    self.sess.run(init)
    self.saver = tf.train.Saver()

      
  
  def train(self):
    """
    """

    ##train the network
    for it in range(self.iterations):

      # sanity check/ debugging of weight changes
      #GRU_disc_weight = self.sess.run(self.GRUdisc_weight)
      #FC_disc_weight  = self.sess.run(self.FCdisc_weight)
      #GRU_gen_weight = self.sess.run(self.GRUgen_weight)
      #FC_gen_weight  = self.sess.run(self.FCgen_weight)      

      # get input batch of real sequence
      batch_ind  = np.random.choice(self.ninput, self.batch_size, replace=False)
      input_batch = self.real_patterns[ batch_ind]

      input_charge = self.charge[ batch_ind  ]
      input_ptInv  = self.ptInv[ batch_ind]
      input_phi    = self.phi[ batch_ind]      


      # generate fake ssid sequences
      rand_latent = np.random.rand( self.batch_size, self.seq_len , self.latent_dim )*2 -1

      rand_charge  = np.array( [ [(2*np.random.randint( 0, 2, 1)-1)[0]]  * self.seq_len for i in range(self.batch_size) ])
      rand_ptInv   = np.array( [ [np.random.uniform(0,1,1)[0]]           * self.seq_len for i in range(self.batch_size) ])
      rand_phi     = np.array( [ [np.random.uniform(0,1,1)[0]]           * self.seq_len for i in range(self.batch_size) ])      
      #rand_ptInv   = np.array( [ [np.random.uniform(0.01,0.5,1)[0]]                           * self.seq_len for i in range(self.batch_size) ])
      #rand_phi     = np.array( [ [np.random.uniform(self.phiRange[0],self.phiRange[1],1)[0]]  * self.seq_len for i in range(self.batch_size) ])      

      input_aux = []
      rand_aux = []
      if "charge" in self.aux_info:
        input_aux.append(input_charge )        
        rand_aux.append(rand_charge)
      if "ptInv" in self.aux_info:
        input_aux.append(input_ptInv)
        rand_aux.append(rand_ptInv)
      if "phi" in self.aux_info:
        input_aux.append(input_phi)
        rand_aux.append(rand_phi)


      input_aux = np.dstack( input_aux)
      rand_aux    = np.dstack( rand_aux)

      gen_seq    = self.sess.run(self.generator, feed_dict={ self.gen_latent_input_placeholder: rand_latent,
                                                             self.gen_aux_input_placeholder:    rand_aux })


      # (1) train critic
      _, self.disc_score, self.disc_cost, \
      self.disc_charge_cost, self.disc_ptInv_cost, self.disc_phi_cost  = self.sess.run([self.disc_train_op, self.scores["disc"], self.costs["disc"],
                                                                                        self.costs["disc_real_aux_charge"], self.costs["disc_real_aux_ptInv"], self.costs["disc_real_aux_phi"]],
                                                                                       feed_dict = { self.disc_real_input_placeholder:  input_batch,
                                                                                                     self.disc_aux_input_placeholder:   input_aux,
                                                                                                     self.gen_latent_input_placeholder: rand_latent,
                                                                                                     self.gen_aux_input_placeholder:    rand_aux })
      
      # (2) train generator
      _, self.gen_score,  self.gen_cost, \
      self.gen_charge_cost, self.gen_ptInv_cost, self.gen_phi_cost     =  self.sess.run([self.gen_train_op, self.scores["gen"],  self.costs["gen"],
                                                                                         self.costs["disc_fake_aux_charge"], self.costs["disc_fake_aux_ptInv"], self.costs["disc_fake_aux_phi"]],
                                                                                        feed_dict={ self.gen_latent_input_placeholder: rand_latent,
                                                                                                    self.gen_aux_input_placeholder:    rand_aux })
      

      # check gradient size
      self.disc_grads = np.max( self.sess.run( self.grads["disc"], feed_dict={self.disc_real_input_placeholder:  input_batch}))

      self.gen_grads  = np.max( self.sess.run( self.grads["gen"],  feed_dict={self.gen_latent_input_placeholder: rand_latent,
                                                                              self.gen_aux_input_placeholder:    rand_aux}))


      if it % self.config.mon_frequency==0:
        self.record_summary(it)
      
      if it%1000 ==0:
        print "iteration:", it,
        print gen_seq[0]
        
      
  def run(self):
    """
    """
    self.initialize()
    self.train()
    self.saver.save(self.sess, self.config.output_path+"model.ckpt")
    self.file_writer.close()


  def generatePatterns(self):

    patternsBankToLoad = {}
    patternsBankToLoad["phi_low"]      = self.phiRange[0]
    patternsBankToLoad["phi_high"]     = self.phiRange[1]
    patternBank = {}
    patternBankTrigSeq = {}
    discontinuity = []
    
    # initialize
    self.initialize()
    self.saver.restore(self.sess, self.config.output_path+"model.ckpt")
    print "nbatch to predict", self.output_bank_size / self.batch_size 
    
    for ibatch in range( self.output_bank_size / self.batch_size   ):
      if ibatch % 100==0:
        print ibatch

      rand_latent = np.random.rand( self.batch_size, self.seq_len , self.latent_dim )*2 -1

      rand_charge  = np.array( [ [(2*np.random.randint( 0, 2, 1)-1)[0]]                        * self.seq_len for i in range(self.batch_size) ])
      rand_ptInv   = np.array( [ [np.random.uniform(0,1,1)[0]]           * self.seq_len for i in range(self.batch_size) ])
      rand_phi     = np.array( [ [np.random.uniform(0,1,1)[0]]           * self.seq_len for i in range(self.batch_size) ])      
      
      #rand_ptInv   = np.array( [ [np.random.uniform(0.01,0.5,1)[0]]                            * self.seq_len for i in range(self.batch_size) ])
      #rand_phi     = np.array( [ [np.random.uniform(self.phiRange[0],self.phiRange[1],1)[0]]   * self.seq_len for i in range(self.batch_size) ])      

      rand_aux = []
      if "charge" in self.aux_info:
        rand_aux.append(rand_charge)
      if "ptInv" in self.aux_info:
        rand_aux.append(rand_ptInv)
      if "phi" in self.aux_info:
        rand_aux.append(rand_phi)
      
      rand_aux    = np.dstack( rand_aux)

      gen_rad, gen_seq, gen_scores = self.sess.run([self.generator_rad, self.generator, self.disc_fake],
                                                   feed_dict={ self.gen_latent_input_placeholder: rand_latent,
                                                               self.gen_aux_input_placeholder: rand_aux })

      for ipat in range(self.batch_size):
        
        thisSSIDs = []
        thisdiscontinuity = 0

        for ilayer in range(len(self.nSSperLayer)):
          thisSSID = (gen_rad[ipat][ilayer]-self.phiRange[0]) // self.phiRes[ilayer]
          thisSSIDs.append( thisSSID )
          if ilayer>0:
            thisdiscontinuity += abs(thisSSIDs[ilayer] -thisSSIDs[ilayer-1] )
            

        thisPatternID = patternID(thisSSIDs, self.nSSperLayer)

        if thisPatternID not in patternBank:
            patternBank[thisPatternID] = 0
            patternBankTrigSeq [ thisPatternID ] = gen_seq[ipat]

        patternBank[thisPatternID] += 1
        discontinuity.append( thisdiscontinuity[0] )
      
    patternsBankToLoad["bank"]         = patternBank
    patternsBankToLoad["bankTrigSeq"]  = patternBankTrigSeq

    print sum(discontinuity)/len(discontinuity)


    with open('PatternBanks/BankGan.pickle', 'wb') as handle:
        pickle.dump(patternsBankToLoad, handle, protocol=pickle.HIGHEST_PROTOCOL)

    
if __name__ == '__main__':

  # train model
  config = config
  model = FTKGan(config)
  model.run()
  model.generatePatterns()
