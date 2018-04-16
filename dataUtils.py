from __future__ import print_function, division

import os
import os.path
import pandas as pd
from io import StringIO
import io
import unicodedata
import re

import tensorflow as tf
import numpy as np
import collections
import random


from numpy import array
from numpy import argmax
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from nltk import sent_tokenize


def load_emb_glove(fp):
    glove_vocab = []
    #glove_embd=[]
    embedding_dict = {}

    file = open(fp,'r')
    for line in file.readlines():
        row = line.strip().split(' ')
        vocab_word = row[0]
        glove_vocab.append(vocab_word)
        embed_vector = [float(i) for i in row[1:]] # convert to list of float
        embedding_dict[vocab_word]=embed_vector
    file.close()

    print('Loaded GLOVE')
    return glove_vocab, embedding_dict, embed_vector

def read_data(raw_text):
    content = raw_text
    content = content.split() #splits the text by spaces (default split character)
    content = np.array(content)
    content = np.reshape(content, [-1, ])
    return content

def build_dictionaries(words, n_max = 200):
    count = collections.Counter(words).most_common(n_max-2) #creates list of word/count pairs;
    dictionary = dict()
    dictionary['PAD'] = len(dictionary)
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    dictionary['START'] = len(dictionary)
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    for word, _ in count:
        dictionary[word] = len(dictionary) #len(dictionary) increases each iteration
        reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    dictionary['UNK'] = len(dictionary)
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, reverse_dictionary

def read_test_train(glove_vocab, embedding_dict, embedding_dim, SEQUENCE_LEN = 50, SEQUENCE_LEN_D = 25, SEQUENCE_LEN_L = 250, dname = 'dict', tr = 0.8, max_vocab = 5000):   
    #read training data
    #six class labels; grades per label in l1-l6
    #df = pd.read_csv('current_data/train_set_all.csv')
    df = pd.read_csv('data/review_text.csv')
    df = df.sample(frac = 1.0)
    text = df['text']

    #80/20 validation split
    split = int(len(df)*tr)
    df_train, df_val = df[:split], df[split:]

    text_val = df_val['text']
    text_train = df_train['text']
    rank_val = df_val['stars']
    rank_train = df_train['stars']

    
    text_all = ' '.join(str(elem) for elem in text)
    training_data = read_data(text_all)


    #Create dictionary and reverse dictionary with word ids
    dictionary, reverse_dictionary = build_dictionaries(training_data, max_vocab)
    print('dictionary len ', len(dictionary))
    
    #save embedding dict
    import csv 
    dict_name = 'data/dict_' + dname + '.csv'    
    w = csv.writer(open(dict_name,'w'))
    for key,val in dictionary.items():
        w.writerow([key,val])
    

    #Create embedding array
    doc_vocab_size = len(dictionary) 
    #dict_as_list = sorted(dictionary.items(), key = lambda x : x[1])

    embeddings_tmp=[]
    c_g = 0
    c = 0


    for i in range(doc_vocab_size):
        item = reverse_dictionary[i]
        if item in glove_vocab:
            embeddings_tmp.append(embedding_dict[item])
            c_g = c_g + 1
        else:
            rand_num = np.random.uniform(low=-0.2, high=0.2,size=embedding_dim)
            embeddings_tmp.append(rand_num)
            c = c + 1

    # final embedding array corresponds to dictionary of words in the document
    embedding = np.asarray(embeddings_tmp)
    print('embedding size:' , embedding.shape[0]) 
    print('Pre-trained Embeddings:', c_g)

    target_val = np.array(rank_val)
    target_train = np.array(rank_train)

    onehot_encoder = OneHotEncoder(sparse=False)
    
    integer_encoded = target_train.reshape(len(target_train), 1)
    y_train = onehot_encoder.fit_transform(integer_encoded)
    
    integer_encoded_val = target_val.reshape(len(target_val), 1)
    y_val = onehot_encoder.fit_transform(integer_encoded_val)
    
    #train set 
    count_oov_train = 0
    count_iv_train = 0
    X_train = []

    for i in df_train['text']:
        i = sent_tokenize(i)
        for j in i[:SEQUENCE_LEN_D]:
            x = read_data(str(j).lower())
            data = []
            data.append(dictionary['START'])
            for word in x:
                if word in dictionary:
                    index = dictionary[word]
                    count_iv_train += 1

                else:
                    index = doc_vocab_size - 1   # dictionary['UNK']
                    count_oov_train += 1

                data.append(index)
            X_train.append(data)
        for k in range(max(SEQUENCE_LEN_D - len(i), 0)):
            X_train.append([0])


    print('OOV in training set: ', count_oov_train)
    print('IV in training set: ', count_iv_train)


 
    #val set
    X_val = []

    count_oov_test = 0
    count_iv_test = 0

    for i in df_val['text']:
        i = sent_tokenize(i)
        for j in i[:SEQUENCE_LEN_D]:
            x = read_data(str(j).lower())
            data = []
            data.append(dictionary['START'])
            for word in x:
                if word in dictionary:
                    index = dictionary[word]
                    count_iv_test += 1

                else:
                    index = doc_vocab_size - 1   # dictionary['UNK']
                    count_oov_test += 1

                data.append(index)
            X_val.append(data)
        for k in range(max(SEQUENCE_LEN_D - len(i), 0)):
            X_val.append([0])

    print('OOV in validation set: ', count_oov_test)
    print('IV in validation set: ', count_iv_test)

    print('len of validation set: ', len(X_val)//SEQUENCE_LEN_D)




    return X_train, y_train, X_val, y_val, doc_vocab_size, embedding

