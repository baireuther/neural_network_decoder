# Readme

This software implements a decoder for small stabilizer codes, in particular for color codes and surface codes, based on a combination of recurrent and feedforward neural networks. The neural networks are implemented using the TensorFlow library [1]. The algorithm is discussed in [2, 3].

The version that is discussed in Ref. [2] is on branch arXiv1705p07855.

The version that is discussed in Ref. [3] is on branch arXiv1804p02926.

## How to use the code to train and evaluate a decoder as discussed in Ref. [3]
1) Generate SQLite databases with your data according to the specifications in data_format.md.<br>
2) Adjust the notebook training.ipynb to your needs and run the training pipeline. Optionally, monitor the training progress using TensorBoard.<br>
3) After the training is completed, adjust and run the notebook evaluation.ipynb to evaluate the trained decoder on a your dataset.


## References
[1] M. Abadi, A. Agarwal, P. Barham, et al, "TensorFlow: Large-scale machine learning on heterogeneous systems" (2015). Software available from tensorflow.org.

[2] P. Baireuther, T. E. O’Brien, B. Tarasinski, and C. W. J. Beenakker, "Machine-learning-assisted correction of correlated qubit errors in a topological code", Quantum 2, 48 (2018).

[3] P. Baireuther, M. D. Caio, B. Criger, C. W. J. Beenakker, and T. E. O’Brien, "Neural network decoder for topological color codes with circuit level noise", arXiv:1804.02926.


