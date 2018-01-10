# Readme for decoder.py

The source code in decoder.py describes a decoder for stabilizer codes, in particular for the surface code. The decoder is implemented using the TensorFlow library [1].

The version that describes the decoder discussed in Ref. [2] is on branch arXiv1705p07855.


## How to use the code to train and evaluate a decoder as discussed in Ref. [2]
1) In the main function of decoder.py, set the path (db_path) to the directory that contains the databases with the training, validation and test data.  
2) In the main function of decoder.py, set the file names of the databases
   WARNING: if the databases do not exist, empty databases will be created at the specified location!
2) In the main function of decoder.py, set the checkpoint_path where feedback and the trained networks will be stored.
   WARNING: The file names of the checkpoints are created automatically.
   Make sure that the target directory is empty because existing files may be overwritten.  
3) In the main function of decoder.py, the function dec.benchmark returns a dictionary containing the fidelities, the logical error rate, standard deviations from bootstrapping, and a few more things. To output the fidelities and the corresponding cycle numbers one could for example add the following two lines at the end of the main function in decoder.py:  
   print("cycles", stats_dict["steps"])  
   print("fidelities", stats_dict["fids"])
4) To run the code: python decoder.py


## References
[1] M. Abadi, A. Agarwal, P. Barham, E. Brevdo, Z. Chen, C. Citro, G. S.Corrado, A. Davis, J. Dean, M. Devin, S. Ghemawat, I. Goodfellow, A. Harp, G. Irving, M. Isard, Y. Jia, R. Jozefowicz, L. Kaiser, M. Kudlur, J. Levenberg, D. Mane, R. Monga, S. Moore, D. Murray, C. Olah, M. Schuster, J. Shlens, B. Steiner, I. Sutskever, K. Talwar, P. Tucker, V. Vanhoucke, V. Vasudevan, F. Vi ́egas, O. Vinyals, P. Warden, M. Wattenberg, M. Wicke, Y. Yu, and X. Zheng, "TensorFlow: Large-Scale Machine Learning on Heterogeneous Systems", arXiv:1603.04467 (2016)  
[2] P. Baireuther, T. E. O'Brien, B. Tarasinski, C. W. J. Beenakker, arXiv:1705.07855 (2017)  
