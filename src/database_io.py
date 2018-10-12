# ========================================================================
# This file is part of a decoder for small stabilizer codes, in particular
# color and surface codes, based on a combination of recurrent and
# feedforward neural networks.
#
# Copyright 2017-2018 Paul Baireuther. All Rights Reserved.
# ========================================================================


# # # LIBRARIES # # #

# Third party libraries
import os
import sqlite3
import numpy as np


# # # CLASSES # # #

class Data:

  """ This class handles the data.

  The __init__ function takes the following parameters:

  training_fname -- Directory + file name of training database.

  validation_fname -- Directory + file name of validation database.

  test_fname -- Directory + file name of test database.

  verbose -- If True, there will be more feedback.

  store_in_memory -- If True the databases will be stored in memory.
  """

  # # # Initialization functions # # #
  def __init__(self, training_fname, validation_fname, test_fname,
               verbose=False, store_in_memory=False):
    """ This function initializes an instance of the Data class. Its inputs
        are described in the class documentation. """

    # Flag to indicate if more feedback is desired
    self.verbose = verbose

    # A dictionary with all databases
    self.db_dict = {}

    # Loading the databases
    if training_fname and os.path.isfile(training_fname):
      self.load_database(training_fname, 'training', store_in_memory)
    if validation_fname and os.path.isfile(validation_fname):
      self.load_database(validation_fname, 'validation', store_in_memory)
    if test_fname and os.path.isfile(test_fname):
      self.load_database(test_fname, 'test', store_in_memory)

  def load_database(self, fname, db_name, store_in_memory=False):
    """ This function loads the databases

    Input
    -----
    fname -- Directory + file name of database
    db_name -- Describes the type of data in the database. Allowed values are
               'train', 'validation', and 'test'.
    store_in_memory -- If True the databases will be stored in memory.
    """

    # Loading the database
    conn = sqlite3.connect(fname)  # open database
    self.db_dict[db_name] = conn  # add database to dictionary
    cur = conn.cursor()  # generate a cursor
    if store_in_memory:
      # Optionally, store database in memory
      cur.execute("PRAGMA temp_store = 2")

    # Counting the number of database entries
    cur.execute('SELECT seed FROM data')
    N_seeds = len(cur.fetchall())
    if self.verbose:
      print("loaded database for " + str(db_name) +
            " with " + str(N_seeds) + " entries.")

  def close_databases(self):
    """ This function closes all databases """
    for conn in self.db_dict.values():
      conn.close()

  def gen_batches(self, n_batches, batch_size, db_type, len_buffered,
                  len_min=None, len_max=None, step_list=None,
                  select_random=True):
    """ This function returns a generator with training, validation, or test
        data batches.

    Input
    -----
    n_batches -- Number of batches.

    batch_size -- Number of samples per batch.

    db_type -- Possible values are "training", "validation", and "test".

    len_buffered -- Number of cycles in the dataset, including "zeros"
                    that may be used as a buffer to the sequences.

    len_min -- If not None, the generated batches will all be of length
               len_min or longer.

    len_max -- If not None, the generated batches will all be of length
               len_max or shorter.

    step_list -- None, or list of integers. If not None, in the case that the
                 db_type is "test", this list determines the lengths of
                 sequences generated during oversampling.

    select_random -- If True, the entries of the database are selected in
                     random order.

    Output
    ------
    A generator containing formatted batches of: seeds, syndrome increments,
    final syndrome increments, number of cycles, and final parities.
    """

    # Check if db_type has one of the allowed values
    if db_type not in ["training", "validation", "test"]:
      raise ValueError(
          "possible db_types are 'training', 'validation', and 'test'.")

    # In case of the "test" db_type, the flag oversample is set
    if db_type == "test":
      oversample = True
    else:
      oversample = False

    # Getting the cursor from the chosen database
    try:
      cur = self.db_dict[db_type].cursor()
    except:
      raise ValueError("The database " + str(db_type) + " does not exist")

    # Making sure that either both len_min and len_max are None, or both are
    # not None
    if (len_min is None and len_max is not None) \
       or (len_min is not None and len_max is None):
      raise ValueError(
          "Either both lenmin, and lenmax, must be set ore none")

    # Selecting data from the database
    if not oversample:
      if len_min is None:
        if select_random:
          cur.execute("SELECT seed, syndrome_increments, final_syndr_incr, " +
                      "parity_of_bitflips, no_cycles " +
                      "FROM data ORDER BY RANDOM() LIMIT ?",
                      (n_batches * batch_size, ))
        else:
          cur.execute("SELECT seed, syndrome_increments, final_syndr_incr, " +
                      "parity_of_bitflips, no_cycles " +
                      "FROM data LIMIT ?",
                      (n_batches * batch_size, ))
      else:
        lmin = len_min
        lmax = len_max
        if select_random:
          cur.execute("SELECT seed, syndrome_increments, final_syndr_incr, " +
                      "parity_of_bitflips, no_cycles " +
                      "FROM data WHERE no_cycles BETWEEN ? and ? " +
                      "ORDER BY RANDOM() LIMIT ?",
                      (lmin, lmax, n_batches * batch_size))
        else:
          cur.execute("SELECT seed, syndrome_increments, final_syndr_incr, " +
                      "parity_of_bitflips, no_cycles " +
                      "FROM data WHERE no_cycles BETWEEN ? and ? LIMIT ?",
                      (lmin, lmax, n_batches * batch_size))
    else:
      cur.execute("SELECT seed, syndrome_increments, final_syndr_incr, " +
                  "parity_of_bitflips FROM data LIMIT ?",
                  (n_batches * batch_size, ))
      if step_list is None:
        step_list = []
        stepsize, step = 1, 1
        for n in range(1, len_max + 1):
          if np.mod(step, stepsize) == stepsize - 1:
            step_list.append(n)
            stepsize += 1
            step = 0
          else:
            step += 1

    # Reshape the data
    for n in range(n_batches):
      arrS, arrX, arrFX, arrL, arrY = [], [], [], [], []
      # Extract data from database
      samples = cur.fetchmany(batch_size)
      for sample in samples:
        if oversample:
          seed_l, syndr_incr_l, fsyndr_incr_l, len_l, parity_l \
              = self._gen_batch_oversample(sample, len_max, step_list)
          # Unfold data
          for seed, syndr_incr, fsyndr_incr, no_cycles, parity in \
                  zip(seed_l, syndr_incr_l, fsyndr_incr_l, len_l, parity_l):
            arrS.append(seed)
            arrX.append(syndr_incr)
            arrFX.append(fsyndr_incr)
            arrL.append(no_cycles)
            arrY.append(parity)
        else:
          seed, syndr_incr, fsyndr_incr, no_cycles, parity \
              = self._gen_batch(sample, len_buffered, len_max)
          arrS.append(seed)
          arrX.append(syndr_incr)
          arrFX.append(fsyndr_incr)
          arrL.append(no_cycles)
          arrY.append(parity)

      arrS = np.array(arrS)
      arrX = np.array(arrX)
      arrFX = np.array(arrFX)
      arrL = np.array(arrL)
      arrY = np.array(arrY).reshape([-1, 1])

      yield arrS, arrX, arrFX, arrL, arrY

  def _gen_batch(self, sample, len_buffered, len_max):
    """ This function formats a single batch of data.

    Input
    -----
    sample - Raw data from the database.

    len_buffered -- Maximum number of cycles in the dataset, including "zeros"
                    that may be used as a buffer to the sequences.

    len_max -- If not None, the generated batches, including the buffer of
               zeroth, will all be of length len_max, or shorter.
    """

    seed, syndr_incr, fsyndr_incr, parity, no_cycles = sample
    len_syndr_incr = int(len(syndr_incr) / len_buffered)

    # Format into shape [steps, syndromes]
    syndr_incr = np.fromstring(syndr_incr, dtype=bool).reshape([
        len_buffered, len_syndr_incr])
    syndr_incr = syndr_incr[:len_max]

    fsyndr_incr = np.fromstring(fsyndr_incr, dtype=bool)
    parity = np.frombuffer(parity, dtype=bool)

    return seed, syndr_incr, fsyndr_incr, no_cycles, parity

  def _gen_batch_oversample(self, sample, len_max, step_list):
    """
    This function formats a single batch of data with final syndrome
    increments and parity at each time step into multiple batches with
    a signle final syndrome increment and parity.

    Input
    -----
    sample - Raw data from the database.

    len_max -- If not None, the generated batches, including the buffer of
               zeroth, will all be of length len_max, or shorter.

    step_list -- A list of cycle numbers, for which batches will be returned.
    """

    seed, syndr_incrs, fsyndr_incrs, parities = sample
    max_steps = min([len(parities), len_max])

    syndr_incrs = np.fromstring(
        syndr_incrs, dtype=bool).reshape([len(parities), -1])
    fsyndr_incrs = np.fromstring(
        fsyndr_incrs, dtype=bool).reshape([len(parities), -1])
    parities = np.frombuffer(parities, dtype=bool)

    dim_syndr_incr = np.shape(syndr_incrs)[1]

    seed_l, syndr_incr_l, fsyndr_incr_l, len_l, parity_l = [], [], [], [], []

    for n in step_list:
      if n <= max_steps:

        seed_l.append(seed)

        # Format into shape [steps, syndromes]
        syndr_incr = np.concatenate(
            (syndr_incrs[:n], np.zeros((max_steps - n, dim_syndr_incr),
                                       dtype=bool)), axis=0)
        syndr_incr_l.append(syndr_incr)

        # Get and set length information
        no_cycles = n
        len_l.append(no_cycles)

        fsyndr_incr_l.append(fsyndr_incrs[n - 1])
        parity_l.append([parities[n - 1]])

    return seed_l, syndr_incr_l, fsyndr_incr_l, len_l, parity_l
