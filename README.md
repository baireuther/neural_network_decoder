# Readme

A decoder for small topological surface and color codes, and potentially other stabilizer codes, when encoding a single logical qubit. The decoder is based on a combination of recurrent and feedforward neural networks. The neural networks are implemented using the TensorFlow library [1]. The algorithm and training procedure are discussed in [2, 3]. Note that, unlike more conventional algorithms, this decoder needs to be trained. Even after training, there is no way to know for certain if the decoder will perform accurately or at all. All one can do is is gather empirical evidence by testing it on a separate test dataset.

The version that is discussed in Ref. [2] is on branch arXiv1705p07855.

The version that is discussed in Ref. [3] is on branch arXiv1804p02926.


## How the code was used in Ref. [3] to train and evaluate a decoder
1) First SQLite databases with data according to the specifications in data_format.md were generated. Suitable pre-processing of the input data was crucial and depends on the quantum circuit and is described in [3].<br>
2) Then notebooks of the form training.ipynb were used to run the training pipeline. The training progress was monitored using TensorBoard.<br>
3) After the training was completed, notebooks of the form evaluation.ipynb were used to evaluate the trained decoder on the test datasets.


## References
[1] M. Abadi, A. Agarwal, P. Barham, et al, "TensorFlow: Large-scale machine learning on heterogeneous systems" (2015). Software available from tensorflow.org.

[2] P. Baireuther, T. E. O’Brien, B. Tarasinski, and C. W. J. Beenakker, "Machine-learning-assisted correction of correlated qubit errors in a topological code", Quantum 2, 48 (2018).

[3] P. Baireuther, M. D. Caio, B. Criger, C. W. J. Beenakker, and T. E. O’Brien, "Neural network decoder for topological color codes with circuit level noise", arXiv:1804.02926.


