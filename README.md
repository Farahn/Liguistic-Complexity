# Liguistic-Complexity

Extension to hierarchical attention network for computing text difficulty. The model uses attention across sentences in a text ("bi-directional context with attention"), and ordinal regression. The original implementation was from https://github.com/ilivans/tf-rnn-attention, modified to use word and sentence level structure, and ordinal instead of logistic regression. 

trained_bca and trained_bca_new are the iPython notebook for running pre-trained models for computing text difficulty. trained_han is for loading and running the pre-trained hierarchical attention network for estimating linguistic complexity. The corresponding pretrained models can be obtained from https://www.dropbox.com/sh/ik2pnukue00g9ay/AADl5gmsqsC-si8_9w72-vXDa?dl=0. Two test sets are also available at https://sites.google.com/site/nadeemf0755/research/linguistic-complexity. The CCS test set can be tested with any of the models, the first test_set should be tested with trained_bca_new, since it has no training data overlapping with the test_set. 

All models use TensorFlow version 1.4
