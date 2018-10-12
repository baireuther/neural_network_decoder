# Notes on the datasets

## Data format for neural network decoder
The datasets are stored in sqlite3 databases, containing a table “data” (mandatory) and a table “info” (optional). Usually, there will be three separate databases for training, validation, and testing. Usually, the training dataset will contain sequences between 1 and 40 cycles, and the validation and test datasets between 1 and 10000 cycles. The expected logical fidelity after the last cycle should not be less than 0.6, otherwise the final parity is too random.

### 1) Training and validation datasets (data table)
Let N<sub>max</sub> be the maximum number of error correction cycles in a given dataset, N<sub>syndr</sub> the size of the syndrome increments, and N<sub>fsyndr</sub> the size of the final syndrome increments.

The table "data" should have the following columns:

* seed: an integer
* no_cycles: an integer
* syndrome_increments: numpy array of size N<sub>max</sub> x N_<sub>syndr</sub> with datatype bool. If the number of cycles is shorter than N<sub>max</sub> the array is buffered with zeros.
* final_syndr_incr: numpy array of size N<sub>fsyndr</sub> with datatype bool.
* parity_of_bitflips: numpy array of size 1 with datatype bool.


### 2) Test datasets (data table)
In the test datasets, all datapoints have the same maximum number of cycles N<sub>max</sub>. Let N<sub>syndr</sub> be the size of the syndrome increments and N<sub>fsyndr</sub> the size of the final syndrome increments.

The table "data" should have the following columns:

* seed: an integer
* syndrome_increments: numpy array of size N<sub>max</sub> x N<sub>syndr</sub> with datatype bool.
* final_syndr_incr: numpy array of size N<sub>max</sub> x N<sub>fsyndr</sub> with datatype bool, where each row in the array contains the final syndrome increment of the cycle corresponding to the row index (assuming the first row index is 1).
* parity_of_bitflips: numpy array of size N<sub>max</sub> with datatype bool, where each row in the array contains the parity of logical bitflips (in the measurement basis) of the cycle corresponding to the row index (assuming the first row index is 1).

### 3) Info table
The table "info" is optional and is meant for information about the dataset. Typical entries would be: Git version number of the error model (“git_version”), error rate per step (“p_step”), and error rate per cycle (“p_cycle”).