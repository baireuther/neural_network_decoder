# Copyright 2017 Paul Baireuther. All Rights Reserved.
#
# ====================================================

"""
A decoder for stabilizer codes, in particular for the surface code, based on
a combination of recurrent and feedforward neural networks. The neural networks
are implemented using the TensorFlow library [1].

References
----------
[1] M. Abadi, A. Agarwal, P. Barham, E. Brevdo, Z. Chen, C. Citro,
    G. S.Corrado, A. Davis, J. Dean, M. Devin, S. Ghemawat, I. Goodfellow,
    A. Harp, G. Irving, M. Isard, Y. Jia, R. Jozefowicz, L. Kaiser, M. Kudlur,
    J. Levenberg, D. Mane, R. Monga, S. Moore, D. Murray, C. Olah, M. Schuster,
    J. Shlens, B. Steiner, I. Sutskever, K. Talwar, P. Tucker, V. Vanhoucke,
    V. Vasudevan, F. Vi ́egas, O. Vinyals, P. Warden, M. Wattenberg, M. Wicke,
    Y. Yu, and X. Zheng, "TensorFlow: Large-Scale Machine Learning on
    Heterogeneous Systems", arXiv:1603.04467 (2016).
"""

import copy
import sqlite3
import numpy as np
import scipy.optimize as optim
import tensorflow as tf
from tensorflow.contrib import rnn


class Decoder:

  """ this class describes two neural networks which consists of several
  long-short term memory (LSTM) layers followed by a feed forward network.
  It is designed for quantum error correction on surface codes but should
  also work on similar codes.

  the __init__ function takes the following parameters:

  dim_syndr: the dimension of the input vector (this vector describes the
             syndrome increments (events))

  dim_fsyndr: the dimension of the final syndrome incements (error signal),
             which are concatenated with the output vector of the last LSTM
             layer (they contain the relevant information from the final
             data-qubit readout)

  lstm1_iss: a list containing the sizes of the internal states of the
             LSTM 1 layers

  lstm2_iss: a list containing the sizes of the internal states of the
             LSTM 2 layers

  ff1_layer_sizes: a list containing the number of neurons per layer of
                   the first feedforward network

  ff2_layer_sizes: a list containing the number of neurons per layer of
                   the second feedforward network

  n_steps_net1: number of syndrome increment steps that constitute the input
                to the first network

  n_steps_net2: the last n_steps_net2 steps of each sequence are fed into the
                second network

  checkpoint_path: a path to which the network gets saved both for
                   intermediate and final versions

  l2w_coeff: coefficient that relates the l2 norm of the weights of the
           feedforward networks to a cost (regularization)

  l2b_coeff: coefficient that relates the l2 norm of the biases of the
           feedforward networks to a cost (regularization)

  lstm_keep_prob: dropout regularization of the LSTM layers, the parameter
                  controlls the percentage of how many entries of the
                  output vectors of the LSTM layers are NOT set to zero
                  during training

  training_phase: this parameter controlls the training phase. in phase 0
  both networks are trained simultaneously. in phase 1 only the first
  network is trained, in phase 2 only the second network is trained.

  learning_rate: the initial learning rate of the Adam optimizer

  seed: currently seeding is not supported
  """

  # # # initialization functions # # #

  def __init__(self, dim_syndr, dim_fsyndr,
               lstm1_iss, lstm2_iss, ff1_layer_sizes, ff2_layer_sizes,
               n_steps_net1, n_steps_net2, checkpoint_path,
               l2w_coeff=0, l2b_coeff=0, lstm_keep_prob=1,
               training_phase=0, learning_rate=0.001, seed=0):
    """ This function initializes an instance of the decoder class, its
    inputs are described in the class documentation """

    # set checkpoint directory (must excist)
    self.cp_path = checkpoint_path

    # init data related variables
    self._init_data_params(dim_syndr, dim_fsyndr)

    # set parameters that define the network size
    self._init_network_params(lstm1_iss, lstm2_iss, ff1_layer_sizes,
                              ff2_layer_sizes, n_steps_net1, n_steps_net2)

    # set parameters that control the training
    self._init_training_params(l2w_coeff, l2b_coeff, lstm_keep_prob,
                               training_phase)

    # build the graph
    self._init_graph()

  def _init_data_params(self, dim_syndr, dim_fsyndr):
    """ a subfunction of __init__, setting variables related to the input
        data, the input variables are described in class description """
    self.dim_syndr = dim_syndr
    self.dim_fsyndr = dim_fsyndr
    self.n_data_qubits = dim_syndr + 1
    self.n_qubits = 2 * dim_syndr + 1

  def _init_network_params(self, lstm1_iss, lstm2_iss, ff1_layer_sizes,
                           ff2_layer_sizes, n_steps_net1, n_steps_net2):
    """ a subfunction of __init__, setting the variables that define the
        network size, the input variables are described in class
        description """

    # LSTM network parameters
    self.lstm1_iss = lstm1_iss
    self.lstm2_iss = lstm2_iss

    # feedfoward network parameters
    self.ff2_lr_s = ff1_layer_sizes
    self.ff1_lr_s = ff2_layer_sizes
    self.n_steps_net1 = n_steps_net1
    self.n_steps_net2 = n_steps_net2

  def _init_training_params(self, l2w_coeff, l2b_coeff, lstm_keep_prob,
                            training_phase):
    """ a subfunction of __init__, setting the variables that define the
        training procedure, the input variables are described in class
        description """

    # regularization of the weights and biases in the feed forward network
    self.l2w_coeff = l2w_coeff
    self.l2b_coeff = l2b_coeff

    # dropout of the outputs in the LSTM network
    self.lstm_keep_prob = lstm_keep_prob

    # setting the training phase and keeping track of trainin process
    self.phase = training_phase
    self.total_trained_epochs = 0
    self.total_trained_batches = 0

  def _init_graph(self):
    """ a subfunction of __init__, defining the graph, initializing the
        tensorflow variables, and the optimizer """

    self.graph = tf.Graph()
    with self.graph.as_default():
      self._init_network_variables()
      self._init_network_functions()

  def _init_network_variables(self):
    """ a subfunction of _init_graph, defining the tensorflow placeholders,
        weights and biases of the feed forward networks and corresponding
        saver instances """

    # placeholders #
    # ... for the input of the syndrome increments
    self.x1 = tf.placeholder(
        "float", [None, self.n_steps_net1, self.dim_syndr])
    self.x2 = tf.placeholder(
        "float", [None, self.n_steps_net2, self.dim_syndr])
    # ... for the input of the final syndrome increments
    self.fx = tf.placeholder(
        "float", [None, self.dim_fsyndr])
    # ... for the parity of the bitflips
    self.y = tf.placeholder(
        "float", [None, 1])
    # ... for the length of dynamic recurrent networks
    self.l1 = tf.placeholder(
        "int32", [None])
    self.l2 = tf.placeholder(
        "int32", [None])

    # ... for the learning rate
    self.lr = tf.placeholder(tf.float32)

    # ... for the dropout (actually keep) probabilities
    self.kp = tf.placeholder(tf.float32)

    # ... for the weight regularization coefficient
    self.l2wc = tf.placeholder(tf.float32)

    # ... for the bias regularization coefficient
    self.l2bc = tf.placeholder(tf.float32)

    # feedfoward network 1 (without final syndrome increment) #
    self.ff1_lrs = self._define_ff_variables(self.lstm1_iss[-1], 1,
                                             self.ff1_lr_s, 'ff1')

    # feedforward network 2 (with final syndrome increment) #
    self.ff2_lrs = self._define_ff_variables(
        self.lstm2_iss[-1] + self.dim_fsyndr, 1, self.ff2_lr_s, 'ff2')

    # define savers for the feedforward network variables
    ff1_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, "ff1")
    self.ff1_saver = tf.train.Saver(ff1_vars)
    ff2_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, "ff2")
    self.ff2_saver = tf.train.Saver(ff2_vars)

  def _define_ff_variables(self, dim_input, dim_output, ff_lr_s, scope):
    with tf.variable_scope(scope):
      ff_lrs = []
      # we add an extra feedforWard layer as the output layer
      n_ff_lrs = len(ff_lr_s) + 1
      for l in range(n_ff_lrs):
        # beginning and output layer are different
        if l == 0:
          m = dim_input
          n = ff_lr_s[l]
        elif l < n_ff_lrs - 1:
          m = ff_lr_s[l - 1]
          n = ff_lr_s[l]
        else:
          m = ff_lr_s[l - 1]
          n = dim_output
        ff_l = {'weights': tf.Variable(
            tf.random_uniform([m, n], minval=-np.sqrt(6 / (m + n)),
                              maxval=np.sqrt(6 / (m + n)))),
                'biases': tf.Variable(tf.zeros([n]))}
        ff_lrs.append(ff_l)
    return ff_lrs

  def _init_network_functions(self):

    out1 = self.network(self.x1)
    out2 = self.network_fsyndr(self.x2, self.fx)
    out1 = tf.reshape(out1, [1, -1, 1])
    out2 = tf.reshape(out2, [1, -1, 1])
    o1po2 = tf.add(out1, out2)
    zeros = tf.zeros_like(o1po2)
    nom = tf.concat([out1, out2], axis=0)
    denom = tf.concat([zeros, o1po2], axis=0)

    logits = tf.reduce_logsumexp(nom, 0) \
        - tf.reduce_logsumexp(denom, 0)

    # probabilities
    self.predictions = tf.nn.sigmoid(logits)

    # cross entropy
    cross_entropy = tf.losses.sigmoid_cross_entropy(
        logits=logits,
        multi_class_labels=tf.reshape(self.y, [-1, 1]))
    self.cost_crossentro = tf.reduce_sum(cross_entropy)

    # add L2 regularization to feedforward layers weights
    self.cost_l2w, self.cost_l2b = 0, 0
    for lr in self.ff1_lrs + self.ff2_lrs:
      self.cost_l2w += self.l2wc * tf.nn.l2_loss(lr['weights'])
      self.cost_l2b += self.l2bc * tf.nn.l2_loss(lr['biases'])

    # cost function
    self.cost = self.cost_crossentro + self.cost_l2w + self.cost_l2b

    # network optimization
    if self.phase == 0:
      self.optimizer = tf.train.AdamOptimizer(
          learning_rate=self.lr).minimize(self.cost)
    elif self.phase == 1:
      self.optimizer = tf.train.AdamOptimizer(
          learning_rate=self.lr).minimize(
              self.cost, var_list=tf.get_collection(
                  tf.GraphKeys.GLOBAL_VARIABLES,
                  "lstm1") + tf.get_collection(
                  tf.GraphKeys.GLOBAL_VARIABLES,
                  "ff1"))
    elif self.phase == 2:
      self.optimizer = tf.train.AdamOptimizer(
          learning_rate=self.lr).minimize(
              self.cost, var_list=tf.get_collection(
                  tf.GraphKeys.GLOBAL_VARIABLES, "lstm2") +
          tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "ff2"))
    else:
      raise ValueError("only training phases 0,1 and 2 are defined")

    # Initializing the variables
    self.initialize_NN = tf.global_variables_initializer()

  def start_session(self, model_file=None, gpu_options=None):
    """ this function (re-)initializes the tensorflow session,
        if a model_file is specified it loads the model's parameters """
    if gpu_options is None:
      self.sess = tf.Session(graph=self.graph)
    else:
      self.sess = tf.Session(graph=self.graph, config=tf.ConfigProto(
          gpu_options=gpu_options))
    self.sess.run(self.initialize_NN)
    # load a model
    if model_file is not None:
      saver = tf.train.Saver()
      saver.restore(self.sess, model_file)

  # # # getter and setter functions # # #
  def set_learning_rate(self, learning_rate):
    """ a function setting the learning rate to be used by the optimizer
        """
    self.learning_rate = learning_rate

  def change_network_length(self, n_steps_net1, n_steps_net2):
    """ function to adjust the maximum number of steps per network """
    # save the all trainable variables
    self.save_variables(self.cp_path + 'temp_change_network_')

    # change the number of steps of the two networks
    self.n_steps_net1 = n_steps_net1
    self.n_steps_net2 = n_steps_net2

    # build the new graph
    self._init_graph()

    # start a new session and restore the variables
    self.sess = tf.Session(graph=self.graph)
    self.sess.run(self.initialize_NN)
    self.load_variables(self.cp_path + 'temp_change_network_')

  def save_network(self, fname):
    """ function to save the entire network

    Input
    -----
    fname -- path + filename to which the network is saved
    """
    saver = tf.train.Saver()
    saver.save(fname)

  def load_network(self, fname):
    """ function to load the entire network

    Input
    -----
    fname -- path + filename from which the network is loaded
    """
    saver = tf.train.Saver()
    saver = tf.train.Saver()
    saver.restore(self.sess, fname)

  # # # functions that load the databases and generate batches # # #

  def load_data(self, training_fname, validation_fname, test_fname):
    """ this function loads the training, validation, and test databases

    Input
    -----

    training_fname -- path + filename of the training database
    valiation_fname -- path + filename of the validation database
    test_fname -- path + filename of the test database
    store_in_RAM -- if True the databases will be stored in the RAM
    """

    self.training_conn = sqlite3.connect(training_fname)
    self.validation_conn = sqlite3.connect(validation_fname)
    self.test_conn = sqlite3.connect(test_fname)

    training_c = self.training_conn.cursor()
    validation_c = self.validation_conn.cursor()
    test_c = self.test_conn.cursor()

    # get all the seeds
    training_c.execute('SELECT seed FROM data')
    validation_c.execute('SELECT seed FROM data')
    test_c.execute('SELECT seed FROM data')
    self.training_keys = list(
        sorted([s[0] for s in training_c.fetchall()]))
    self.validation_keys = list(
        sorted([s[0] for s in validation_c.fetchall()]))
    self.test_keys = list(sorted([s[0] for s in test_c.fetchall()]))

    # checks that there is no overlapp in the seeds of the data sets
    N_training = len(self.training_keys)
    N_validation = len(self.validation_keys)
    N_test = len(self.test_keys)
    all_keys = set(self.training_keys +
                   self.validation_keys + self.test_keys)
    if len(all_keys) < N_training + N_validation + N_test:
      raise ValueError("There is overlapp between the seeds of the"
                       " training,  validation, and test sets. This"
                       "is bad practice")
    print("loaded databases and checked exclusiveness training, "
          "validation, and test keys")

    print("N_training=" + str(N_training) + ", N_validaiton=" +
          str(N_validation) + ", N_test=" + str(N_test) + ".")

  def close_databases(self):
    """ This function closes all databases """
    self.training_conn.close()
    self.validation_conn.close()
    self.test_conn.close()

  def gen_batch(self, sample):
    """ formats a single batch of data

    Input
    -----

    sample - raw data from the database
    """

    syndr, fsyndr, parity, length = sample
    n_steps = int(len(syndr) / self.dim_syndr)

    # format into shape [steps, syndromes]
    syndr1 = np.fromstring(syndr, dtype=bool).reshape([n_steps, -1])

    # get and set length information
    len1 = np.frombuffer(length, dtype=int)[0]

    # the second length is set by n_steps_net2, except if len1 is shorter
    len2 = min(len1, self.n_steps_net2)

    syndr2 = syndr1[len1 - len2:len1 - len2 + self.n_steps_net2]
    fsyndr = np.fromstring(fsyndr, dtype=bool)
    parity = np.frombuffer(parity, dtype=bool)

    return syndr1, syndr2, fsyndr, len1, len2, parity

  def gen_batch_oversample(self, sample, max_steps=None):
    """ formats a single batch of data with final syndrome increments
        at each time steps into multiple batches with a signle final
        syndrome increment

    Input
    -----

    sample - raw data from the database
    max_steps -- maximum number of steps for oversampling
    """

    syndrs, fsyndrs, parities = sample
    max_steps = min([len(parities), max_steps])

    syndrs = np.fromstring(syndrs, dtype=bool).reshape([len(parities), -1])
    fsyndrs = np.fromstring(
        fsyndrs, dtype=bool).reshape([len(parities), -1])
    parities = np.frombuffer(parities, dtype=bool)

    syndr1_l, syndr2_l, fsyndr_l, len1_l, len2_l, parity_l \
        = [], [], [], [], [], []

    step_list = []
    stepsize, step = 1, 1
    for n in range(self.n_steps_net2, max_steps + 1):
      if np.mod(step, stepsize) == stepsize - 1:
        step_list.append(n)
        stepsize += 1
        step = 0
      else:
        step += 1

    for n in step_list:
      if n <= max_steps:

        # format into shape [steps, syndromes]
        syndr1 = np.concatenate(
            (syndrs[:n], np.zeros((max_steps - n, self.dim_syndr),
                                  dtype=bool)), axis=0)
        syndr1_l.append(syndr1)

        # get and set length information
        len1 = n
        len1_l.append(len1)

        # the second length is set by n_steps_net2, except if len1 is
        # shorter
        len2 = min(len1, self.n_steps_net2)
        len2_l.append(len2)

        syndr2_l.append(
            syndr1[len1 - len2:len1 - len2 + self.n_steps_net2])
        fsyndr_l.append(fsyndrs[n - 1])
        parity_l.append(parities[n - 1])

    return syndr1_l, syndr2_l, fsyndr_l, len1_l, len2_l, parity_l

  def gen_batches(self, batch_size, n_batches, data_type, oversample=False,
                  max_steps=None):
    """ a genererator to generate the training, validation, and test
        batches

    Input
    -----

    batch_size -- number of samples per batch
    n_batches -- number of batches
    data_type -- 'training', 'validation', or 'test'  data

    Output
    ------

    a generator containing formatted batches of:
    syndrome increments for the first network, for the second network,
    the final syndrome increments, length information for the first
    network, for the second network, and the final parities
    """

    # select data from the corresponding database
    if data_type == "training":
      c = self.training_conn.cursor()
    elif data_type == "validation":
      c = self.validation_conn.cursor()
    elif data_type == "test":
      c = self.test_conn.cursor()
    else:
      raise ValueError("The only allowed data_types are: 'training', "
                       "'validation' and 'test'.")
    if oversample:
      c.execute("SELECT events, err_signal, parities " +
                "FROM data ORDER BY RANDOM() LIMIT ?",
                (n_batches * batch_size, ))
    else:
      c.execute("SELECT events, err_signal, parity, length " +
                "FROM data ORDER BY RANDOM() LIMIT ?",
                (n_batches * batch_size, ))

    for n in range(n_batches):

      arrX1, arrX2, arrFX, arrL1, arrL2, arrY = [], [], [], [], [], []
      samples = c.fetchmany(batch_size)

      for sample in samples:
        if oversample:
          syndr1_l, syndr2_l, fsyndr_l, len1_l, len2_l, parity_l \
              = self.gen_batch_oversample(sample, max_steps)

          for syndr1, syndr2, fsyndr, len1, len2, parity in \
                  zip(syndr1_l, syndr2_l, fsyndr_l, len1_l, len2_l, parity_l):
            arrX1.append(syndr1)
            arrX2.append(syndr2)
            arrFX.append(fsyndr)
            arrL1.append(len1)
            arrL2.append(len2)
            arrY.append(parity)
        else:
          syndr1, syndr2, fsyndr, len1, len2, parity \
              = self.gen_batch(sample)

          arrX1.append(syndr1)
          arrX2.append(syndr2)
          arrFX.append(fsyndr)
          arrL1.append(len1)
          arrL2.append(len2)
          arrY.append(parity)

      arrX1 = np.array(arrX1)
      arrX2 = np.array(arrX2)
      arrFX = np.array(arrFX)
      arrL1 = np.array(arrL1)
      arrL2 = np.array(arrL2)
      arrY = np.array(arrY)

      yield arrX1, arrX2, arrFX, arrL1, arrL2, arrY

  # # # functions that define the neural networks # # #

  def network(self, input_syndr):
    """ This function defines the first neural network, that does NOT
        get the final syndrome increments """

    # LSTM 1 #
    # create LSTM cells with optional dropout
    cells = []
    i = 0
    for iss in self.lstm1_iss:
      cell = rnn.BasicLSTMCell(iss)
      cell = rnn.DropoutWrapper(cell=cell, output_keep_prob=self.kp)
      cells.append(cell)
      i += 1

    # combine LSTM cells
    lstm_cells = rnn.MultiRNNCell(cells)
    # define a recurrent network of length self.l1
    lstm_out, lstm_states = tf.nn.dynamic_rnn(
        lstm_cells, input_syndr,
        sequence_length=self.l1,
        dtype=tf.float32, scope="lstm1")

    # get the output of the last LSTM cell after the last time-step
    last_lstm_out = tf.gather_nd(lstm_out,
                                 tf.stack([tf.range(tf.shape(lstm_out)[0]),
                                           self.l1 - 1], axis=1))
    # an optional activation layer after the last LSTM
    last_lstm_out = tf.nn.relu(last_lstm_out)

    # for training feedback we gather the activations of the last LSTM
    # layer
    self.lstm1_activation = last_lstm_out

    # define a saver for the variables of LSTM 1
    lstm_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, "lstm1")
    self.lstm1_saver = tf.train.Saver(lstm_vars)

    # FEEDFORWARD NETWORK 1 #

    # for training feedback we gather the activations of each feedforward
    # layer
    self.ff1_activations = []

    # the input of the first feedforward layer is the output of the
    # last LSTM cell after the last time-step self.l1
    ff_net = last_lstm_out

    # build the feedforward network
    for ff_l in self.ff1_lrs[:-1]:
      ff_net = tf.matmul(ff_net, ff_l['weights']) + ff_l['biases']
      ff_net = tf.nn.relu(ff_net)
      self.ff1_activations.append(ff_net)
      ff_net = tf.nn.dropout(ff_net, self.kp)

    # the output layer is linear (i.e. not activated)
    output = tf.matmul(ff_net, self.ff1_lrs[-1]['weights']) \
        + self.ff1_lrs[-1]['biases']
    return output

  def network_fsyndr(self, input_syndr, input_fsyndr):
    """ This function defines the second neural network, that does
        get the final syndrome increments """

    # LSTM 2 #
    # create LSTM cells with optional dropout
    cells = []
    i = 0
    for iss in self.lstm2_iss:
      cell = rnn.BasicLSTMCell(iss)
      cell = rnn.DropoutWrapper(cell=cell, output_keep_prob=self.kp)
      cells.append(cell)
      i += 1

    # combine LSTM cells
    lstm_cells = rnn.MultiRNNCell(cells)
    # define a recurrent network of length self.l2
    lstm_out, lstm_states = tf.nn.dynamic_rnn(
        lstm_cells, input_syndr,
        sequence_length=self.l2,
        dtype=tf.float32, scope="lstm2")

    # get the output of the last LSTM cell after the last time-step
    last_lstm_out = tf.gather_nd(lstm_out,
                                 tf.stack([tf.range(tf.shape(lstm_out)[0]),
                                           self.l2 - 1], axis=1))
    # an optional activation layer after the last LSTM
    last_lstm_out = tf.nn.relu(last_lstm_out)

    # for training feedback we gather the activations of the last LSTM
    # layer
    self.lstm2_activation = last_lstm_out

    # define a saver for the variables of LSTM 2
    lstm_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, "lstm2")
    self.lstm2_saver = tf.train.Saver(lstm_vars)

    # FEEDFORWARD NETWORK 1 #

    # for training feedback we gather the activations of each feedforward
    # layer
    self.ff2_activations = []

    # the input of the first feedforward layer is the output of the
    # last LSTM cell after the last time-step self.l1 concatenated
    # with the final syndrome increment
    ff_net = tf.concat([last_lstm_out, input_fsyndr], 1)

    # build the feedforward network
    for ff_l in self.ff2_lrs[:-1]:
      ff_net = tf.matmul(ff_net, ff_l['weights']) + ff_l['biases']
      ff_net = tf.nn.relu(ff_net)
      self.ff2_activations.append(ff_net)
      ff_net = tf.nn.dropout(ff_net, self.kp)

    # the output layer is linear (i.e. not activated)
    output = tf.matmul(ff_net, self.ff2_lrs[-1]['weights']) \
        + self.ff2_lrs[-1]['biases']
    return output

  # # # functions for training the network # # #

  def train_one_epoch(self, mini_batch_size=32, learning_rate=0.001,
                      max_validation_batch_size=10**6, max_batches=10**9):
    """ this function controlls the training of the networks

    Input
    -----
    mini_batch_size: size of mini batches used for the stochastic gradient
                     descent
    learning_rate: learning rate of adam optimizer

    Output
    ------
    self.total_trained_epochs -- number of epochs trained on
    self.total_trained_batches -- number of mini batches trained on
    cost_ce_training -- the average cross_entropy of the training data
    cost_l2w -- the average cost from weight regularization
    cost_l2b -- the average cost from bias regularization
    plog_training -- logical error rate of training data
    plog_validation -- logical error rate of validation data
    plog_SD_training -- error bars of plog_training
    plog_SD_validaiton -- error bars of plog_validation
    activations -- the average of the activations for each layer
    """

    # set the learning rate of the adam optimizer
    self.set_learning_rate(learning_rate)

    # generate an epoch of training data
    # how many mini batches are available per epoch, and how many to use
    n_batches = min(int(len(self.training_keys) /
                        mini_batch_size), max_batches)
    batch_iter = self.gen_batches(
        mini_batch_size, n_batches, data_type='training')

    # training the network
    for i in range(n_batches):

      # prepare the training data
      batch_x1, batch_x2, batch_fx, batch_l1, batch_l2, batch_y = next(
          batch_iter)

      # do the actual training
      self.sess.run(self.optimizer,
                    feed_dict={self.x1: batch_x1,
                               self.x2: batch_x2,
                               self.fx: batch_fx,
                               self.l1: batch_l1,
                               self.l2: batch_l2,
                               self.y: batch_y,
                               self.lr: self.learning_rate,
                               self.l2wc: self.l2w_coeff,
                               self.l2bc: self.l2b_coeff,
                               self.kp: self.lstm_keep_prob})
      self.total_trained_batches += 1
    self.total_trained_epochs += 1

    # generate some training feedback after each epoch
    batch_sz = min(len(self.validation_keys), max_validation_batch_size)
    # costs
    cost_ce_training, cost_l2w, cost_l2b = self._calc_cost(
        batch_sz, data_type='training')
    # cost_ce_validation, cost_l2w_validation, cost_l2b_validation = self._calc_cost(
    #     batch_sz, data_type='validation')
    # logical error rates and activations
    plog_training, plog_validation, activations\
        = self._calc_feedback(batch_sz, verbose=True)

    # note that cost_l2 is batch independent
    return self.total_trained_epochs, self.total_trained_batches, \
        cost_ce_training, cost_l2w, cost_l2b, \
        plog_training, plog_validation, activations

  # # # training routine # # #

  def do_training(self, io_fname, prefix="", batch_size=64,
                  max_wait=10, learning_rate=0.001,
                  max_batches_per_epoch=10**9,
                  max_validation_batch_size=10**9,
                  plog_min=0.5, feedback_step=1, reset_file=False):
    """
    Input
    -----
    io_fname -- filename for feedback, it will be stored at the
                self.cp_path + io_fname
    max_batches_per_epoch -- number of batches per training epoch
    batch_size -- number of examples in each batch
    max_validation_batch_size -- size of the (single) batch that is used from
                                 validation during training
    learning_rate -- the learning rate used in the Adam optimizer
    reset_file -- if True, the feedback file will be overwritten
    max_wait -- the training stops after the performance on the validation
                file has not improved for max_wait epochs
    """

    # load data from file
    fb = load_feedback(self.cp_path, io_fname, reset_file)
    try:
      self.total_trained_epochs = fb["epochs"][-1]
      self.total_trained_batches = fb["batches"][-1]
      if plog_min > min(fb["plogs_validation"]):
        plog_min = min(fb["plogs_validation"])
        print("restored previous min(plog validation) of " +
              "{:.2f}".format(plog_min * 100) + "%.")
    except:
      self.total_trained_epochs = 0
      self.total_trained_batches = 0

    # mcount keeps track of rounds without improvement
    mcount = 0
    while(mcount < max_wait):

      # train for one epoch
      ret = self.train_one_epoch(batch_size, learning_rate,
                                 max_batches=max_batches_per_epoch,
                                 max_validation_batch_size=max_validation_batch_size)
      n_epochs, plog_validation = ret[0], ret[6]
      fb = update_feedback(fb, ret)

      # check if result sets a new record, and if yes, save network
      if plog_validation < plog_min:
        plog_min = plog_validation
        self.save_variables(path=self.cp_path + prefix + 'best_')
        print("\n The result improved after " + str(mcount + 1) +
              " epochs to " + str(plog_validation * 100) + "%\n")
        mcount = 0
      else:
        mcount += 1

      # print feedback and save results
      if np.mod(n_epochs, feedback_step) == 0:
        save_feedback(fb, self.cp_path + io_fname)

    # in the end 1st save the final network and then
    # load the record holding network config
    self.save_variables(path=self.cp_path + prefix + 'final_')

    print("The best logical error rate on the validation dataset was " +
          str(plog_min * 100), "%")

  # # # functions related to training feedback # # #

  def _calc_cost(self, batch_size, data_type):
    """ this function calculates the costs of a given batch

    Output
    ------
    cost_ce -- cross entropy
    cost_l2 -- regularization cost
    """

    # generate batch
    batch_iter_feedback = self.gen_batches(batch_size, 1, data_type)
    batch_x1, batch_x2, batch_fx, batch_l1, batch_l2, batch_y \
        = next(batch_iter_feedback)

    fd = {self.x1: batch_x1, self.x2: batch_x2, self.fx: batch_fx,
          self.l1: batch_l1, self.l2: batch_l2, self.y: batch_y,
          self.l2wc: self.l2w_coeff, self.l2bc: self.l2b_coeff, self.kp: 1}

    # calculate cross entropy
    cost_ce = self.cost_crossentro.eval(feed_dict=fd, session=self.sess)

    # calculate regularization cost
    cost_l2w = self.cost_l2w.eval(feed_dict=fd, session=self.sess)
    cost_l2b = self.cost_l2b.eval(feed_dict=fd, session=self.sess)

    return cost_ce, cost_l2w, cost_l2b

  def get_activations(self, batch_x1, batch_x2, batch_fx, batch_l1, batch_l2,
                      batch_y):
    """ This function calculates the activations for all layers

    Output
    ------
    activations -- the average activations of the network layers in
                   the following order LSTM 1, LSTM 2, feedforward 1,
                   feedforward 2
    """

    activations = []
    for layer in [self.lstm1_activation, self.lstm2_activation] \
            + self.ff1_activations + self.ff2_activations:

      units = self.sess.run(layer,
                            feed_dict={self.x1: batch_x1,
                                       self.x2: batch_x2,
                                       self.fx: batch_fx,
                                       self.l1: batch_l1,
                                       self.l2: batch_l2,
                                       self.kp: 1})
      activations.append(np.mean(np.reshape(units, [-1])))
    return activations

  def _calc_feedback(self, batch_sz, bootstrap=False, verbose=False):
    """
    Input
    -----
    batch_sz -- size of batch
    verbose -- print the feedback

    Output
    ------
    plog_training -- logical error rate of training data
    plog_validation -- logical error rate of validation data
    plog_SD_training -- error bars of plog_training
    plog_SD_validation -- error bars of plog_validation
    activations -- the average activations of the network layers in
                   the following order LSTM 1, LSTM 2, feedforward 1,
                   feedforward 2
    """

    # # # TRAINING DATA # # #

    # generate a batch of training data
    batch_iter_feedback = self.gen_batches(
        batch_sz, 1, data_type='training')
    batch_x1, batch_x2, batch_fx, batch_l1, batch_l2, batch_y \
        = next(batch_iter_feedback)

    # calculate the logical error rate for the training batch
    stats_dict_training = self.calc_fids_and_plog(
        batch_x1, batch_x2, batch_fx, batch_l1, batch_l2, batch_y,
        bootstrap, x0_max=0.01)
    plog_training = stats_dict_training['plog']

    # average layer activations
    activations = self.get_activations(batch_x1, batch_x2, batch_fx,
                                       batch_l1, batch_l2, batch_y)

    # # # VALIDATION DATA # # #

    # generate a batch of validation data
    batch_iter_feedback = self.gen_batches(
        batch_sz, 1, data_type='validation')
    batch_x1, batch_x2, batch_fx, batch_l1, batch_l2, batch_y = next(
        batch_iter_feedback)

    # in general the validation runs can have more steps. if this is the
    # case we need to adjust the length of the network
    net1_steps, max_steps = self.n_steps_net1, max(batch_l1)
    if max_steps > net1_steps:
      self.change_network_length(max_steps, self.n_steps_net2)

    stats_dict_validation = self.calc_fids_and_plog(
        batch_x1, batch_x2, batch_fx, batch_l1, batch_l2, batch_y,
        bootstrap, x0_max=0.01)

    # change the network length back (if it was changed)
    if max_steps > net1_steps:
      self.change_network_length(net1_steps, self.n_steps_net2)
    plog_validation = stats_dict_validation['plog']

    if verbose:
      print("Trained for " + str(self.total_trained_epochs) + " epochs (" +
            str(round(int(self.total_trained_batches) / 1000)) +
            "k mini_batches). Training logical p is " +
            "{:.2f}".format(plog_training * 100) + "%. " +
            "Validation logical p is " +
            "{:.2f}".format(plog_validation * 100) + "%.")

    return plog_training, plog_validation, activations

  # # # functions to evaluate the network # # #

  def calc_fidelity(self, batch_size, data_type, bootstrap=True, fname=None):

    # generate batch
    batch_iter = self.gen_batches(batch_size, 1, data_type)
    batch_x1, batch_x2, batch_fx, batch_l1, batch_l2, \
        batch_y = next(batch_iter)
    stats_dict = self.calc_fids_and_plog(batch_x1, batch_x2, batch_fx,
                                         batch_l1, batch_l2, batch_y,
                                         bootstrap)
    # save data
    if fname is not None:
      self._save_fids(stats_dict, fname)

    return stats_dict

  def calc_fids_and_plog(self, batch_x1, batch_x2, batch_fx, batch_l1,
                         batch_l2, batch_y, bootstrap, x0_max=10):
    """ this function calculates fidelity decay curves,
    it then fits them to the standard fidelity decay
    formula and calculates the logical error rate

    Output
    -------
    p_logical -- logical error rate
    p_logical_SD -- standard deviation of the logical error rate
    fids -- the fidelities at the given steps
    fids_SEM -- standard deviation of the fidelities

    """

    # get probabilities that a bitflip occured from the network
    preds = self.predictions.eval(
        feed_dict={self.x1: batch_x1,
                   self.x2: batch_x2,
                   self.fx: batch_fx,
                   self.l1: batch_l1,
                   self.l2: batch_l2,
                   self.kp: 1},
        session=self.sess)
    preds = np.array(preds)
    preds = np.reshape(preds, [-1])

    # make a decision of the total parity of bitflips
    for m in range(len(preds)):
      if preds[m] < 0.5:
        preds[m] = 0
      else:
        preds[m] = 1
    preds = preds.astype(int)

    # reshape the true results for comparison
    batch_y = batch_y.reshape((-1))

    # compare predictions to true results
    comp_list = np.equal(preds, batch_y).astype('float')

    # reshape into a list of lists
    comparison = []
    for n in range(max(batch_l1)):
      comparison.append([])
    for n in range(len(comp_list)):
      idx = batch_l1[n] - 1
      comparison[idx].append(comp_list[n])

    # use same number of values for each step
    # lmin = min([len(el) for el in comparison])
    # compare_arr = np.array([el[:lmin] for el in comparison])

    # do statistics
    stats_dict = calc_stats(comparison, bootstrap, x0_max)

    return stats_dict

  def benchmark(self, prefix=None, N_max=10**10, N_batches=100,
                max_steps=None, bootstrap=False, bm_fname=None,
                verbose=True, oversample=False, return_raw_data=False):
    """ this function benchmarks the system on the test data. Do NOT use
        this function for early stopping! It first rebuilds the network
        to cope with the generally longer sequences of the test set,
        does a benchmark and then brings the network back to its original
        state.

    Input
    -----
    prefix -- specify a certain checkpoint, such as e.g. "best"
    N_max -- maximum number of sampels to average over
    N_batches -- split the evaluation in N_batches to prevent memory
                     overflow
    bootstrap -- calculate error bars using bootstrapping
    bm_fname -- save the benchmark to this file
    verbose -- print some feedback
    oversample -- decompose a run which has final syndrome increments at
                  each step (unphysical) into multiple runs with a single
                  final syndrome increment in the end

    Output
    ------
    stats_dict -- dictionary containing the benchmark information
    """

    # save the current status of the network
    n_steps_net1_old = self.n_steps_net1
    self.save_variables(self.cp_path + "temp_for_bm_")

    # how many examples will be averaged over?
    N = int(min(len(self.test_keys), N_max) / N_batches) * N_batches
    if verbose:
      print("The benchmark will be over", N, "examples.")

    # make a test batch
    batch_iter = self.gen_batches(
        N, 1, data_type='test', oversample=oversample, max_steps=max_steps)
    batch_x1, batch_x2, batch_fx, batch_l1, batch_l2, batch_y = next(
        batch_iter)

    batch_x1s = np.split(batch_x1, N_batches)
    batch_x2s = np.split(batch_x2, N_batches)
    batch_fxs = np.split(batch_fx, N_batches)
    batch_l1s = np.split(batch_l1, N_batches)
    batch_l2s = np.split(batch_l2, N_batches)

    # if prefix is given, load new parameters
    if prefix is not None:
      self.load_variables(self.cp_path + prefix)

    # change network to accommodate the test runs
    self.change_network_length(len(batch_x1[0]), self.n_steps_net2)

    # evaluate the test batch
    choicesNN, res = [], []
    for n in range(100):
      res += list(self.predictions.eval(feed_dict={
          self.x1: batch_x1s[n],
          self.x2: batch_x2s[n],
          self.fx: batch_fxs[n],
          self.l1: batch_l1s[n],
          self.l2: batch_l2s[n],
          self.kp: 1},
          session=self.sess))
    res = np.array(res)

    # extract the predictions
    preds = np.reshape(res, [-1])
    for m in range(len(preds)):
      if preds[m] < .5:
        preds[m] = 0
      else:
        preds[m] = 1
    choices = preds.astype(int)
    choices = choices.transpose()
    choicesNN = np.array(choices)

    # compare predictions to labels
    batchy_copy = copy.deepcopy(batch_y[:len(choicesNN)])
    batchy_copy = batchy_copy.reshape((-1))
    comp_list = np.equal(choicesNN, batchy_copy).astype('float')

    # reshape the results for easier comparison
    comparison = []
    for n in range(self.n_steps_net1):
      comparison.append([])
    for n in range(len(comp_list)):
      idx = batch_l1[n] - 1
      comparison[idx].append(comp_list[n])

    # use same number of values for each step
    # lmin = min([len(el) for el in comparison])
    # compare_arr = np.array([el[:lmin] for el in comparison])

    # check if all steps have the same number of values
    lmin = min([len(el) for el in comparison])
    lmax = max([len(el) for el in comparison])
    if lmax > lmin:
      print("WARNING: the number of examples per step varies from",
            lmin, "to", lmax)

    # do statistics
    stats_dict = calc_stats(comparison, bootstrap, verbose=verbose)

    # save
    if bm_fname is not None:
      np.save(bm_fname, stats_dict)

    # restore the network to it's state before the benchmark
    self.change_network_length(n_steps_net1_old, self.n_steps_net2)
    self.load_variables(self.cp_path + "temp_for_bm_")

    if return_raw_data:
      return stats_dict, comparison
    else:
      return stats_dict

  # # # functions to save and restore the networks # # #

  def save_variables(self, path):
    if self.phase == 0:
      self.lstm1_saver.save(self.sess, path + "lstm1")
      self.ff1_saver.save(self.sess, path + "ff1")
      self.lstm2_saver.save(self.sess, path + "lstm2")
      self.ff2_saver.save(self.sess, path + "ff2")
    elif self.phase == 1:
      self.lstm1_saver.save(self.sess, path + "lstm1")
      self.ff1_saver.save(self.sess, path + "ff1")
    elif self.phase == 2:
      self.lstm2_saver.save(self.sess, path + "lstm2")
      self.ff2_saver.save(self.sess, path + "ff2")

  def load_variables(self, path):
    try:
      self.lstm1_saver.restore(self.sess, path + "lstm1")
      self.ff1_saver.restore(self.sess, path + "ff1")
    except:
      0  # do nothing

    try:
      self.lstm2_saver.restore(self.sess, path + "lstm2")
      self.ff2_saver.restore(self.sess, path + "ff2")
    except:
      0  # do nothing


# # # functions related to feedback # # #

def load_feedback(path, io_fname, reset_file=False):
  # load data from file
  try:
    feedback = np.loadtxt(path + io_fname)
  except:
    feedback = [[] for i in range(8)]
  if reset_file:
    feedback = [[] for i in range(8)]
  fb = {"epochs": list(feedback[0]),
        "batches": list(feedback[1]),
        "costs_ce_training": list(feedback[2]),
        "costs_l2w": list(feedback[3]),
        "costs_l2b": list(feedback[4]),
        "plogs_training": list(feedback[5]),
        "plogs_validation": list(feedback[6]),
        "activations": [list(el) for el in feedback[7:]]}
  return fb


def update_feedback(fb, ret):
  keys = ["epochs", "batches", "costs_ce_training", "costs_l2w", "costs_l2b",
          "plogs_training", "plogs_validation"]
  for i in range(7):
    fb[keys[i]].append(ret[i])
  for i in range(len(ret[7])):
    try:
      fb["activations"][i].append(ret[7][i])
    except:
      fb["activations"].append([])
      fb["activations"][i].append(ret[7][i])
  return fb


def save_feedback(fb, fname):
  feedback = []
  n_epochs = len(fb["epochs"])
  for key in ["epochs", "batches", "costs_ce_training", "costs_l2w",
              "costs_l2b", "plogs_training", "plogs_validation",
              "activations"]:
    feedback.append(np.reshape(fb[key], [-1, n_epochs]))
  feedback = np.concatenate(feedback, axis=0)
  np.savetxt(fname, feedback)


# # # functions for statistics and visualization # # #

def decay(x, p_logical, x0):
  """ This functions is used to make a exponential fit to the fidelity
      curves """
  return (1 + (1 - 2 * p_logical)**(x - x0)) / 2.


def calc_stats(data, bootstrap, n_sampling=5000, x0_max=10, verbose=False,
               visualize_bootstrapping=False):
  """ calculates the logical error rate and error bars """

  # since it is possible that the batch does not contain fidelities
  # for all steps, hence we need a list with all steps for which
  # predictions excist (we call it 'steps')
  steps, data_nonzero = [], []
  fids, rs_means_l, plogs_bs = [], [], []

  # in the following we assume that the first step is s = 1
  for s in range(1, len(data) + 1):
    dat = data[s - 1]
    if len(dat) != 0:
      # non-trivial data points
      steps.append(s)
      data_nonzero.append(dat)
      # fidelities
      fids.append(np.mean(dat))

  # fit decay curve to the non-tivial data
  popt, pcov = optim.curve_fit(
      decay, steps, fids, bounds=((0.0001, 0.0001), (.1, x0_max)))
  plog, x0 = popt[0], popt[1]
  if x0 > 0.99 * x0_max:
    print("WARNING, x0 is larger than", x0_max,
          "the fitting algorithm fails")
  if plog > .09:
    print("WARNING, plog is larger than 9\%, the fitting algorithm fails")

  # use bootstrapping to calculate error bars
  if bootstrap:
    # bootstrapping for error bars on fidelities and logical p
    # here it is assumed that all steps -- if they exist -- have the same
    # number of entries
    ls = [len(dat) for dat in data_nonzero]
    if min(ls) != max(ls):
      raise ValueError("The array with the fidelities is not uniform")

    # for resampling data_nonzero must be an array
    data_nonzero = np.array(data_nonzero)
    _, n_samples = np.shape(data_nonzero)
    for n in range(n_sampling):
      idcs = np.random.choice(n_samples, size=n_samples)
      rs_data = data_nonzero[:, idcs]
      rs_means = np.array(np.mean(rs_data, axis=1))
      rs_means_l.append(rs_means)
      popt, pcov = optim.curve_fit(decay, np.array(steps), rs_means,
                                   bounds=(0, [.99, 10]))
      plogs_bs.append(popt[0])
    fids_sdv_bs = np.std(rs_means_l, axis=0)
    plog_sdv_bs = np.std(plogs_bs)

  if bootstrap:
    res_dict = {'steps': steps,
                'fids': fids, 'fids_sdv_bs': fids_sdv_bs,
                'plog': plog, 'plog_sdv_bs': plog_sdv_bs, 'x0': x0}
    if verbose:
      print("logical error rate:", round(plog * 100, 5),
            "+-", round(plog_sdv_bs * 100, 5), "%")
      print("x0 offset", round(x0, 3))
  else:
    res_dict = {'steps': steps, 'fids': fids, 'plog': plog, 'x0': x0}
    if verbose:
      print("logical error rate:", round(plog * 100, 5))
      print("x0 offset", round(x0, 3))

  return res_dict


if __name__ == '__main__':

  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
  # Define pathes and names for training, validation, and test databases  #
  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

  """ WARNING: if the data bases do not exist, empty databases will be
      created at the specified destination.
  """
  # path of databases (must exist)
  db_path =

  # filenames of databases (this must be sqlite3 databases)
  train_fname =
  validation_fname =
  test_fname =

  # # # # # # # # # # # # Structure of the databases  # # # # # # # # # #
  # To learn more about the structure of the databases, please consult  #
  # https://github.com/baireuther/circuit_model                         #
  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
  # Define path and names to store checkpoints of the network and         #
  # feedback from training                                                #
  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

  """ WARNING: Make sure that the checkpoint_path directory is empty because
  existing files might be overwritten.
  """
  # path for feedback and checkpoints (must exist and be empty)
  checkpoint_path =
  feedback_fname = "feedback"

  # # # Initialize an instance of the decoder # # #
  dec = Decoder(dim_syndr=8, dim_fsyndr=4,
                lstm1_iss=[64, 64], lstm2_iss=[64, 64],
                ff1_layer_sizes=[64], ff2_layer_sizes=[64],
                n_steps_net1=20, n_steps_net2=3,
                checkpoint_path=checkpoint_path,
                l2w_coeff=10 * 10**(-6), l2b_coeff=0,
                lstm_keep_prob=0.8,
                training_phase=0)
  # start a tensorflow session
  dec.start_session()

  # load the databases
  dec.load_data(db_path + train_fname,
                db_path + validation_fname,
                db_path + test_fname)

  # # # train the network # # #
  dec.do_training(io_fname=feedback_fname,
                  max_batches_per_epoch=10000, batch_size=64,
                  max_validation_batch_size=10000,
                  learning_rate=10 * 10**(-4),
                  reset_file=False, max_wait=100)

  # # # Evaluate the trained decoder on the test dataset # # #
  stats_dict = dec.benchmark("best_", bootstrap=True, oversample=True,
                             N_max=5 * 10**4, N_batches=100, max_steps=300,
                             verbose=True)

  print("DONE")
