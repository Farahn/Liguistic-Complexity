# Liguistic-Complexity

Extension to hierarchical attention network for computing text difficulty. The model uses attention across sentences in a text ("bi-directional context with attention"), and ordinal regression. The original implementation was from https://github.com/ilivans/tf-rnn-attention, modified to use word and sentence level structure, and ordinal instead of logistic regression. 

trained_bca is the iPython notebook for running pre-trained model for computing text difficulty. The pretrained model can be obtained from https://www.dropbox.com/sh/ik2pnukue00g9ay/AADl5gmsqsC-si8_9w72-vXDa?dl=0. A test set is also available at https://sites.google.com/site/nadeemf0755/research/linguistic-complexity.

All models use TensorFlow version 1.4
