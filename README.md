# Liguistic-Complexity

Extension to hierarchical attention network for computing text difficulty from the paper Estimating Linguistic Complexity for Science Texts, Farah Nadeem and Mari Ostendorf, The Thirteenth Workshop on Innovative Use of NLP for Building Educational Applications, NAACL 2018. 

The model uses attention across sentences in a text ("bi-directional context with attention"), and ordinal regression. The original implementation was from https://github.com/ilivans/tf-rnn-attention, modified to use word and sentence level structure, and ordinal instead of logistic regression. 

trained_bca and trained_bca_new are the iPython notebook for running pre-trained models for computing text difficulty. trained_han is for loading and running the pre-trained hierarchical attention network for estimating linguistic complexity. The corresponding pretrained models can be obtained from https://1drv.ms/f/s!Ag4UUgKkf0ZPu55vBR6_-6WOuAO-Ug. Two test sets are also available at https://sites.google.com/site/nadeemf0755/research/linguistic-complexity. The CCS test set can be tested with any of the models. 

Models have been updated to TF 1.8.
