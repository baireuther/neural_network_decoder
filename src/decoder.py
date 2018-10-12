# ========================================================================
# This file is part of a decoder for small stabilizer codes, in particular
# color and surface codes, based on a combination of recurrent and
# feedforward neural networks.
#
# Copyright 2017-2018 Paul Baireuther. All Rights Reserved.
# ========================================================================


# # # LIBRARIES # # #

# Third party libraries
from itertools import tee
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn

# This project
from qec_functions import calc_plog


class Decoder:

  """ This class describes a neural network which consist of several
  long-short term memory (LSTM) layers followed by two heads consitings of
  feed forward layers. It is designed for quantum error correction on
  stabilizer codes, such as the surface codes or the color code.

  The __init__ function takes the following parameters:

  code_distance -- The distance of the quantum error correction code.

  dim_syndr -- The dimension of the syndrome increments, i.e. the number
               of stabilizers. In a surface 17 code for example, it would be 8.

  dim_fsyndr -- The dimension of the final syndrome increment (which one can
                calculate from the readout of the data qubits).

  lstm_iss -- A list containing the sizes of the internal states of the
              LSTM layers.

  ff_layer_sizes -- A list containing the number of neurons per layer of
                    the feedforward networks.

  checkpoint_path -- A path to which the network gets saved to, both for
                     intermediate and final instances.

  keep_prob -- Dropout regularization of the feedforward and LSTM layers. This
               parameter controlls the percentage of how many entries of the
               output vectors are NOT set to zero (on average) during training.

  aux_loss_factor -- This scalar parameters controls how much the cost function
                     weighs the auxillary head of the network. If it is 1,
                     then both heads are weighted equally.

  l2_prefactor -- Prefactor for L2 weight regularization of the feed forward
                  layers.
  """

  # # # Initialization functions # # #

  def __init__(self, code_distance, dim_syndr, dim_fsyndr, lstm_iss,
               ff_layer_sizes, checkpoint_path, keep_prob=1, aux_loss_factor=1,
               l2_prefactor=0):
    """ This function initializes an instance of the Decoder class. Its inputs
        are described in the class documentation above. """

    # Factor by which the loss of the auxillary head is multiplied
    self.aux_loss_factor = aux_loss_factor

    # Directory to save the network and feedback (must exist)
    self.cp_path = checkpoint_path

    # Initialize input related variables
    self._init_data_params(code_distance, dim_syndr, dim_fsyndr)

    # Set parameters that define the network size
    self._init_network_params(lstm_iss, ff_layer_sizes)

    # Set parameters that control the training
    self._init_training_params(keep_prob, l2_prefactor)

    # Build the graph
    self._init_graph()

  def _init_data_params(self, code_distance, dim_syndr, dim_fsyndr):
    """ A subfunction of __init__, setting variables related to the input
        data, the input variables are described in class documentation above.
    """
    self.code_dist = code_distance
    self.dim_syndr = dim_syndr
    self.dim_fsyndr = dim_fsyndr

  def _init_network_params(self, lstm_iss, ff_layer_sizes):
    """ A subfunction of __init__, setting the variables that define the
        network size, the input variables are described in class documentation
        above. """

    # LSTM layers' parameters
    self.lstm_iss = lstm_iss

    # Feedfoward layers' parameters
    self.ff_aux_lr_s = [lstm_iss[-1]] + ff_layer_sizes + [1]
    self.ff_lr_s = [lstm_iss[-1] + self.dim_fsyndr] + ff_layer_sizes + [1]

  def _init_training_params(self, keep_prob, l2_prefactor):
    """ A subfunction of __init__, setting the variables that define the
        training procedure, the input variables are described in class
        documentation above. """

    # Dropout of the outputs in the LSTM network
    self.kp = keep_prob

    # Prefactor for L2 weight regularization (feedforward layers only)
    self.l2_prefact = l2_prefactor

    # Variables to keep track of training process
    self.total_trained_epochs = 0
    self.total_trained_batches = 0

  def _init_graph(self):
    """ A subfunction of __init__, defining the graph, initializing the
        tensorflow variables, and the optimizer. """

    self.graph = tf.Graph()
    with self.graph.as_default():
      self._init_network_variables()
      self._init_network_functions()

  def _init_network_variables(self):
    """ A subfunction of _init_graph, defining the tensorflow placeholders,
        weights and biases of the feed forward networks, and corresponding
        saver instances. """

    # Here we defind placeholders ...
    with tf.variable_scope('input'):
      # ... for the input of the syndrome increments
      self.x = tf.placeholder(tf.float32,
                              [None, None, self.dim_syndr],
                              name='x_input')
      # ... for the input of the final syndrome increments
      self.fx = tf.placeholder(tf.float32, [None, self.dim_fsyndr],
                               name='fx_input')
      # ... for the parity of the bitflips
      self.y = tf.placeholder(tf.float32, [None, 1], name='y_input')
      # ... for the number of stabilizer measurement cycles in a sequence
      self.length = tf.placeholder(tf.int32, [None], name='length_input')

    with tf.variable_scope('training_parameters'):
      # ... for the learning rate
      self.lr = tf.placeholder(tf.float32, name='learning_rate')
      # ... for the weighing of the auxillary head
      self.alf = tf.placeholder(tf.float32, name='aux_loss_factor')

      # ... for the dropout (keep probabilities)
      self.lstm_kp = tf.placeholder(tf.float32, name='lstm_keep_probability')
      self.ff_kp = tf.placeholder(tf.float32, name='ff_keep_probability')

    with tf.variable_scope('summary_placeholders'):
      # ... for the tensorboard summaries
      self.plog = tf.placeholder(tf.float32, name='plog_train')
      self.plog_aux = tf.placeholder(tf.float32, name='plog_aux_train')
      self.tot_cost = tf.placeholder(tf.float32, name='tot_cost')

  def _init_network_functions(self):
    """ A subfunction of _init_graph, defining all the main functions of the
        graph. """

    with tf.variable_scope('NET'):
      # Gathering the network outputs (logits)
      out, out_aux = self.network(self.x, self.fx)
      logits = tf.reshape(out, [-1, 1])
      logits_aux = tf.reshape(out_aux, [-1, 1])

    with tf.variable_scope('prediction'):
      # Calculating the probabilities that the parity of bitflips is odd
      self.predictions = tf.nn.sigmoid(logits)
      self.predictions_aux = tf.nn.sigmoid(logits_aux)
      p = tf.nn.sigmoid(logits)
      p_aux = tf.nn.sigmoid(logits_aux)

      # Adding the network outputs and predictions to the summary
      tf.summary.histogram('logits', clip(logits),
                           collections=['feedback'])
      tf.summary.histogram('p', p, collections=['feedback'])
      tf.summary.histogram('logits_aux', clip(logits_aux),
                           collections=['feedback'])
      tf.summary.histogram('p_aux', p_aux, collections=['feedback'])

    with tf.variable_scope('cost'):
      # Calculate the cross entropy for the main head
      cross_entropy = tf.losses.sigmoid_cross_entropy(
          logits=logits, multi_class_labels=self.y)
      self.cost_crossentro = tf.reduce_sum(cross_entropy)

      # Calculate the cross entropy for the auxillary head
      cross_entropy_aux = tf.losses.sigmoid_cross_entropy(
          logits=logits_aux,
          multi_class_labels=self.y)
      self.cost_crossentro_aux = tf.reduce_sum(cross_entropy_aux)

      # Calculate the L2 norm of the feed forward networks' weights
      # to do weight regularization (not for the biases)
      col_ff = tf.get_collection(
          tf.GraphKeys.TRAINABLE_VARIABLES, "NET/NET_FF")
      weights_l = []
      for el in col_ff:
        if "weights" in el.name:
          weights_l.append(el)
      self.l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in weights_l]) \
          * self.l2_prefact
      # Print some feedback
      print("added the following variables to L2 weight regularization term:")
      for var in weights_l:
        print(var.name)

      # The total cost function is the sum of the two cross-entropies plus
      # the weight regularization term.
      self.cost = self.cost_crossentro + self.l2_loss + \
          self.aux_loss_factor * self.cost_crossentro_aux

    # Writing the costs and logical error rates to the feedback summary.
    with tf.variable_scope('feedback'):
      # costs
      tf.summary.scalar('crossentropy', self.cost_crossentro,
                        collections=['feedback'])
      tf.summary.scalar('crossentropy_aux', self.cost_crossentro_aux,
                        collections=['feedback'])
      tf.summary.scalar('l2_loss', self.l2_loss,
                        collections=['feedback'])
      tf.summary.scalar('cost', self.tot_cost, collections=['feedback'])

      # logical error rate
      tf.summary.scalar('logical_error_rate', self.plog,
                        collections=['feedback'])
      tf.summary.scalar('logical_error_rate_aux', self.plog_aux,
                        collections=['feedback'])

    # Writing some feedback regarding the nature of the input data
    with tf.variable_scope('network_parameters'):
      tf.summary.scalar('min_length', tf.reduce_min(
          self.length), collections=['feedback'])
      tf.summary.scalar('max_length', tf.reduce_max(
          self.length), collections=['feedback'])

    with tf.variable_scope('optimizer'):
      # Defining the network optimization algorithm
      self.optimizer = tf.train.AdamOptimizer(
          learning_rate=self.lr).minimize(self.cost)

    # Tensorflow saver, to save checkpoints of the network
    self.saver = tf.train.Saver()

    # Merge summaries
    self.merged_summaries = tf.summary.merge_all('network')
    self.merged_summaries_fb = tf.summary.merge_all('feedback')

    # Define separate summary writers for training and validation
    self.train_writer = tf.summary.FileWriter(
        self.cp_path + '/tensorboard/training', self.graph)
    self.val_writer = tf.summary.FileWriter(
        self.cp_path + '/tensorboard/validation', self.graph)

    # Finally, we initialize the network variables
    self.initialize_NN = tf.global_variables_initializer()

  # # # Functions that define the neural network # # #
  def network(self, input_syndr, input_fsyndr):
    """ This function defines the neural network.

    Input
    -----
    input_syndr -- A placeholder that will later contain the lists with
                   syndrome increments.

    input_fsyndr -- A placeholder that will later contain the final
                    syndrome increments.


    Output
    ------
    ff_net -- The neural network with the main evaluation head.

    ff_net_aux -- The neural network with the auxillary evaluation head.
    """

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # #  Recurrent part (LSTM) of the network # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    with tf.variable_scope('NET_LSTM'):
      # First, we create LSTM cells with (optional) dropout. Here dropout is
      # only applied to the outputs of the layers.
      cells = []
      for iss in self.lstm_iss:
        cell = rnn.BasicLSTMCell(iss)
        cell = rnn.DropoutWrapper(cell=cell, output_keep_prob=self.lstm_kp)
        cells.append(cell)

      # Next, we combine LSTM cells into a recurrent network.
      lstm_cells = rnn.MultiRNNCell(cells)
      # Since each batch will contain sequences of variying number or
      # measurement cycles, we use dynamic_rrn which allows for a variable
      # sequence_length.
      lstm_out, lstm_states = tf.nn.dynamic_rnn(
          lstm_cells, input_syndr,
          sequence_length=self.length,
          dtype=tf.float32, scope="dyn_rnn")

      # We now select the output of the last LSTM cell after the last cycle, as
      # specified by self.length.
      last_lstm_out = tf.gather_nd(lstm_out,
                                   tf.stack([tf.range(tf.shape(lstm_out)[0]),
                                             self.length - 1], axis=1))
      # We store this last output as feedback to be displayed with tensorboard.
      tf.summary.histogram("lstm_out", clip(
          last_lstm_out), collections=['feedback'])
      # Finally, we apply a rectified linear activation function to the output.
      last_lstm_out = tf.nn.relu(last_lstm_out)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # Feedforward part of the network # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # # # Main evaluation head # # #
    with tf.variable_scope('NET_FF'):
      # The main evaluation head is a feedforward network with rectified linear
      # units. It gets as an input last_lstm_out concatenated with the final
      # syndrome increment.
      ff_net = tf.concat([last_lstm_out, input_fsyndr], 1)

      # Here we define the feedforward network. The output will be a
      # single float number. The final output is not subject to an
      # activation function.
      n_lrs = len(self.ff_lr_s) - 1
      for n in range(n_lrs):
        if n < n_lrs - 1:
          act_fct = tf.nn.relu
          has_bias = True
        else:
          act_fct = tf.identity
          has_bias = False
        ff_net = self._make_layer(ff_net, self.ff_lr_s[n],
                                  self.ff_lr_s[n + 1],
                                  act_fct, "layer_" + str(n), has_bias)

    # # # Auxillary evaluation head # # #
    with tf.variable_scope('NET_FF_aux'):
      # The auxillary evaluation head is a feedforward network with rectified
      # linear units. It gets as an input last_lstm_out. Its purpose is to
      # encourage translation invariance in time of the recurrent network.
      ff_net_aux = last_lstm_out

      # Here we define the feedforward network. The output will be a
      # single float number. The final output is not subject to an
      # activation function.
      n_lrs = len(self.ff_aux_lr_s) - 1
      for n in range(n_lrs):
        if n < n_lrs - 1:
          act_fct = tf.nn.relu
          has_bias = True
        else:
          act_fct = tf.identity
          has_bias = False
        ff_net_aux = self._make_layer(ff_net_aux, self.ff_aux_lr_s[n],
                                      self.ff_aux_lr_s[n + 1],
                                      act_fct, "layer_" + str(n), has_bias)

    return ff_net, ff_net_aux

  def _make_layer(self, input_tensor, dim_in, dim_out, act_fct, name,
                  has_bias=True):
    """ This function builds a single layer of a feedforward network.

    Input
    -----
    input_tensor -- The tensor that will be fed into the layer.

    dim_in -- The dimension of the input tensor.

    dim_out -- The number of neurons in the layer.

    act_fct -- The activation function.

    name -- The name of the layer.

    has_bias -- If True, the layer will have a bias vector added to
                the neurons (before the activation function is applied).


    Output
    ------
    layer_after_dropout -- The layer with dropout.

    """

    with tf.variable_scope(name):
      # First, initialize the weights and biases
      weights = tf.Variable(
          tf.random_uniform([dim_in, dim_out],
                            minval=-1.0 / np.sqrt(dim_in),
                            maxval=+1.0 / np.sqrt(dim_in)),
          dtype=tf.float32, name="weights")
      tf.summary.histogram("weights", clip(
          weights), collections=['network'])
      if has_bias:
        biases = tf.Variable(tf.constant(0.0, tf.float32, [dim_out]),
                             dtype=tf.float32, name="biases")
        tf.summary.histogram("biases", clip(biases),
                             collections=['network'])
      # Second, define the layer.
      if has_bias:
        layer = tf.matmul(input_tensor, weights) + biases
      else:
        layer = tf.matmul(input_tensor, weights)
      # ... and write some feedback
      tf.summary.histogram("preact", clip(layer),
                           collections=['feedback'])

      # Third, apply the activation function
      layer_activations = act_fct(layer, name="activations")
      # ... and write some feedback
      tf.summary.histogram("act", clip(layer_activations),
                           collections=['feedback'])

      # Finally, apply dropout to the output of the layer
      layer_after_dropout = tf.nn.dropout(layer_activations, self.ff_kp)
    return layer_after_dropout

  # # # Starting and ending the session # # #
  def start_session(self, gpu_options=None):
    """ This function (re-)initializes the TensorFlow session """

    # Optionally, GPU options can be specified.
    if gpu_options is None:
      self.sess = tf.Session(graph=self.graph)
    else:
      self.sess = tf.Session(graph=self.graph, config=tf.ConfigProto(
          gpu_options=gpu_options))

    # Initialize the TensorFlow session.
    self.sess.run(self.initialize_NN)

  def end_session(self):
    """ This function ends the current TensorFlow session """
    self.sess.close()

  def feed_dict(self, b_x, b_fx, b_l, b_y, learning_rate,
                lstm_kp, ff_kp):
    """ This function creates and returns a 'feed dictionary', that
    can be passed to the session for network training and evaluation
    purposes

    Input
    -----
    b_x -- A batch with syndrome increments.

    b_fx -- A batch with final syndrome increments.

    b_l -- A batch specifying the number of stabilizer measurement
           cycles for each batch entry.

    b_y -- A batch with the true parities.

    learning_rate -- A single float number which sets the learning
                     rate of the optimizer.

    lstm_kp -- A single float number that sets the keep-rate of the
               dropout after the LSTM layers.

    ff_kp -- A single float number that sets the keep-rate of the
             dropout after the feedforward layers.
    """
    return {self.x: b_x, self.fx: b_fx,
            self.length: b_l, self.y: b_y, self.lr: learning_rate,
            self.lstm_kp: lstm_kp, self.ff_kp: ff_kp}

  # # # Functions for training the network # # #
  def train_one_epoch(self, train_batches, learning_rate=0.001):
    """ This function controlls the training of the network for one epoch.

    Input
    -----
    train_batches -- The batches that consitute one epoch.

    learning_rate -- The learning rate of the (Adam) optimizer.

    """

    # Here we train the network
    for b_s, b_x, b_fx, b_l, b_y in train_batches:
      fd = self.feed_dict(b_x, b_fx, b_l, b_y,
                          learning_rate, self.kp, self.kp)
      self.sess.run(self.optimizer, feed_dict=fd)
      self.total_trained_batches += 1
    self.total_trained_epochs += 1

    # After the training we write some feedback to the summaries
    summary = self.sess.run(self.merged_summaries, feed_dict=fd)
    self.train_writer.add_summary(summary, self.total_trained_epochs)

    # Finally, we save the network
    self.save_network("model")

  def calc_feedback(self, batches, validation=True):
    """ This function calculates feedback and summaries.

    Input
    -----
    batches -- The batches over which the feedback is calculated.

    validation -- If True, the validation summary writer is used.
                  If False, the traning summary writer is used.

    """

    # First, we copy the batch generator two times
    batches, batches_copy = tee(batches)
    batches, batches_copy_aux = tee(batches)

    # We use the first copy to calculate the logical error rate for
    # the main head ...
    plog = calc_plog(self.test_net(batches_copy, auxillary=False))
    # ... and the second copy for the auxillary head
    plog_aux = calc_plog(self.test_net(batches_copy_aux, auxillary=True))

    # We then use the original generator to calculate the cost.
    # For practicality, we only use a single batch for that and
    # the other feedback
    b_s, b_x, b_fx, b_l, b_y = next(batches)
    fd = self.feed_dict(b_x, b_fx, b_l, b_y, 0, 1, 1)
    cost = self.cost.eval(session=self.sess, feed_dict=fd)

    # Finally, we write this information to the summary
    fd = {self.plog: plog, self.plog_aux: plog_aux, self.tot_cost: cost,
          self.x: b_x, self.fx: b_fx,
          self.length: b_l, self.y: b_y, self.lr: 0,
          self.lstm_kp: 1, self.ff_kp: 1}
    summary_fb = self.sess.run(self.merged_summaries_fb, feed_dict=fd)
    if validation:
      self.val_writer.add_summary(summary_fb, self.total_trained_epochs)
      print("logical error rate on validation set is", round(plog, 4))
    else:
      self.train_writer.add_summary(summary_fb, self.total_trained_epochs)
      print("logical error rate on training set is", round(plog, 4))
    return plog

  def test_net(self, batches, auxillary=False):
    """ A function that evaluates the network.

    Input
    -----
    batches -- A generator of batches that will be evaluated.

    auxillary -- If True, the auxillary head is used for evaluation.
                 If False, the main head is used for evaluation.

    """

    length_list, comp_list = [], []
    for b_s, b_x, b_fx, b_l, b_y in batches:
      # Get probabilities that a bitflip occured from the network
      fd = self.feed_dict(b_x, b_fx, b_l, b_y, 0, 1, 1)
      if auxillary:
        preds = self.predictions_aux.eval(feed_dict=fd, session=self.sess)
      else:
        preds = self.predictions.eval(feed_dict=fd, session=self.sess)

      # Reshape tensors to 1D
      preds = np.reshape(preds, [-1])
      b_y = np.reshape(b_y, [-1])

      # Make a decision (0 for even, 1 for odd) of the total parity of bitflips
      preds = np.around(preds).astype(bool)

      # Compare predictions to true results
      comp_list += list(np.equal(preds, b_y))

      # Update maximum length of time-sequence
      length_list += list(b_l)

    # Reshape into a list of lists
    comparison = [[] for _ in range(max(length_list))]
    for n in range(len(comp_list)):
      idx = length_list[n] - 1
      comparison[idx].append(comp_list[n])

    return comparison

  # # # Functions to save and restore the network # # #
  def save_network(self, fname, with_step=True):
    """ Function to save all variables of the network.

    Input
    -----
    fname -- The  filename to which the network is saved, note
             that by default the path is the checkpoint_path/model/.

    with_step -- If True, the variable "global_step" will be set to
                 self.total_trained_epochs. This can be useful if the
                 training has been interrupted.
    """

    with self.graph.as_default():
      saver = tf.train.Saver()
      if with_step:
        saver.save(self.sess, self.cp_path + "model/" + fname,
                   global_step=self.total_trained_epochs)
      else:
        saver.save(self.sess, self.cp_path + "model/" + fname)

  def load_network(self, fname):
    """ Function to load the network from a checkpoint.

    Input
    -----
    fname -- is the filename from which the network is loaded, note
             that by default the path is the checkpoint_path/model/.

    """
    with self.graph.as_default():
      saver = tf.train.Saver()
      saver.restore(self.sess, self.cp_path + "model/" + fname)


# # # Functions # # #

def clip(var):
  """ Clips elements of a tensor at mean +- 3 standard deviations.

  Input
  -----
  var -- A tensorflow tensor.
  """

  lin = tf.reshape(var, [-1])
  mean, variance = tf.nn.moments(lin, axes=[0])
  sigma = tf.sqrt(variance)
  cv_min = mean - 3 * sigma
  cv_max = mean + 3 * sigma
  return tf.clip_by_value(var, cv_min, cv_max)
