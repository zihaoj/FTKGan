import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell
from tensorflow.contrib.layers import fully_connected
import numpy as np
import pickle
from config import config


## input sequence of [shape batch * length of sequence * embedding]

def build_Discriminator_GRU(inputs, scope, reuse = False):
  
  with tf.variable_scope(scope, reuse = reuse):

    num_neurons = 32
    cell = GRUCell(num_neurons)
  
    outputs, states = tf.nn.dynamic_rnn(
      cell,
      inputs,
      dtype=tf.float32
    )

    last = outputs[:, -1, :]

    ## get discriminator score
    prediction = tf.contrib.layers.fully_connected(inputs =last, num_outputs = 1, activation_fn = tf.sigmoid)
    
    return prediction


def build_Generator_GRU(inputs, phiRange, scope, reuse = False):


  with tf.variable_scope(scope, reuse = reuse):
    gen_seq_length = inputs.get_shape()[1]
    
    num_neurons = 32
    cell = GRUCell(num_neurons)
    
    outputs, states = tf.nn.dynamic_rnn(
      cell,
      inputs,
    dtype=tf.float32
    )
    
    output_seq = []
    
    for iq in xrange(gen_seq_length ):
      logits = tf.contrib.layers.fully_connected(inputs =outputs[:, iq, :], num_outputs = 1, activation_fn = tf.sigmoid)
      rad    = logits*( phiRange[1]-phiRange[0]) + phiRange[0]
      
      ## manual embedding with cosine and sine
      out_cos = tf.cos( rad )
      out_sin = tf.sin( rad )

      tmp = tf.concat([ out_cos, out_sin], axis=1)
      tmp = tf.expand_dims(tmp, 1)
      output_seq.append(tmp)

      
    return tf.concat( output_seq , axis = 1)
  


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
    self.lr         = self.config.lr
    self.iterations = self.config.iterations

    self.truth_patterns  = self.config.truth_patterns
    self.phiRange        = self.config.phiRange

    ## get network graphs initialized
    self.add_placeholders()
    self.build_generator_and_discriminator()
    self.add_loss_op()
    self.add_optimizer_op()


    ## Loading banks
    ## Concactenate all banks
    patternSequence = []
    for t in self.truth_patterns:
        tmp = None
        with open(t, 'rb') as handle:
            tmp = pickle.load(handle)

        for p in tmp["bank"].keys():
            patternSequence.append( tmp["bankTrigSeq"][p]  )

    self.real_patterns = np.array( patternSequence )

    
            

  def add_placeholders(self):
    """
    Adds placeholders to the graph
    Set up the observation, action, and advantage placeholder
    """

    self.disc_real_input_placeholder = tf.placeholder(tf.float32, shape=(None, self.seq_len, self.embedsize) )
    self.disc_fake_input_placeholder = tf.placeholder(tf.float32, shape=(None, self.seq_len, self.embedsize) )    

    self.gen_input_placeholder  = tf.placeholder(tf.float32, shape=(None, self.seq_len, self.latent_dim) )

  
  def build_generator_and_discriminator(self):
    """
    """

    self.D = build_Discriminator_GRU#self.disc_input_placeholder, "Discriminator")
    self.G = build_Generator_GRU#self.gen_input_placeholder, "Generator")

  
  def add_loss_op(self):

    self.disc_cost =  - tf.reduce_mean( self.D( self.disc_real_input_placeholder, "Discriminator") ) +\
                      tf.reduce_mean( self.D( self.disc_fake_input_placeholder, "Discriminator", reuse = True ) )


    self.generator = self.G( self.gen_input_placeholder, self.phiRange, "Generator")
    self.gen_cost  = - tf.reduce_mean( self.D( self.disc_fake_input_placeholder, "Discriminator", reuse = True ) )

  
  def add_optimizer_op(self):

    #gen_pars  = [p for p in tf.trainable_variables() if "Generator" in p.name]
    #disc_pars = [p for p in tf.trainable_variables() if "Discriminator" in p.name]
    
    self.gen_train_op  = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.gen_cost) #,  var_list= gen_pars)
    self.disc_train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.disc_cost) #, var_list= disc_pars)


  def initialize(self):
    """
    """

    # create tf session
    self.sess = tf.Session()

    init = tf.global_variables_initializer()
    self.sess.run(init)
    self.saver = tf.train.Saver()

    
  
  
  def train(self):
    """
    """

    ##train the network
    for it in range(self.iterations):

      # generate fake ssid sequences
      rand_latent = np.random.rand( self.batch_size, self.seq_len , 1 )

      gen_seq = self.sess.run(self.generator,
                              feed_dict={ self.gen_input_placeholder: rand_latent })

      # first train critic
      self.sess.run(self.disc_train_op,
                    feed_dict = { self.disc_real_input_placeholder: self.real_patterns[it*self.batch_size: (it+1)*self.batch_size  ],
                                  self.disc_fake_input_placeholder: gen_seq})

      disc_cost =   self.sess.run(self.disc_cost,
                                  feed_dict = { self.disc_real_input_placeholder: self.real_patterns[it*self.batch_size: (it+1)*self.batch_size  ],
                                                self.disc_fake_input_placeholder: gen_seq})


      # then train the generator
      self.sess.run(self.gen_train_op, feed_dict={ self.disc_fake_input_placeholder: gen_seq})

      gen_cost  =   self.sess.run(self.gen_cost, feed_dict={ self.disc_fake_input_placeholder: gen_seq})


      print "disc_cost", disc_cost, "gen_cost", gen_cost
      
      
  def run(self):
    """
    """
    # initialize
    self.initialize()
    self.train()
    self.saver.save(self.sess, self.config.output_path+"model.ckpt")


          
if __name__ == '__main__':

  # train model
  config = config
  model = FTKGan(config)
  model.run()
