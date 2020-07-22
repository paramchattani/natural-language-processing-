# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 18:11:58 2020

@author: param
"""

import nltk
from nltk import word_tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import numpy as np 
import random 
import pickle 
from collections import Counter
lemmatizer=WordNetLemmatizer()
hm_lines=10000000
nltk.download('punkt')
#nltk.download('wordnet')
def create_lexicon(pos,neg):
    lexicon=[]
    for fil in [pos,neg]:
        with open(fil,'r') as f:
            contents=f.readlines()
            #accesing each line one by one using contents
            for l in contents[:hm_lines]:
                #access words using l 
                allwords=word_tokenize(l.lower())
                # suppose sentence 'I am xyz'
                # what tokenize does is - ['I','am','param']
                lexicon+=list(allwords)
                #append in list
    lexicon=[lemmatizer.lemmatize(i) for  i in lexicon]
    # what lemmatizer does is that it seperates meaning ful words from whole lexicon 
    #whole lexicon built
    word_count=Counter(lexicon)
    #format w_counts={'the':5252,'a':5555}
    l2=[]
    for w in word_count:
        if 50<word_count[w]<1000:
            l2.append(w)
    return l2
# we are trying to seperate most common words from our lexicon that do not hold meaning for example 'a', 'the' , etc  
    #final lexicon 
def sample_handling(sample,lexicon,classification):
    featureset=[]
    with open(sample,'r') as f:
        contents=f.readlines()
        #reeading each line by opening line 
        for l in contents[:hm_lines]:
            current_word=word_tokenize(l.lower())
            current_word=[lemmatizer.lemmatize(i) for  i in current_word]
            # final current_words 
            features=np.zeros(len(lexicon))
        # we are trying to take features in one hot method and initializing our features as [0,0,0,0,0,0,0,0,0,0,......]
            for word in current_word:
                if word.lower() in lexicon:
                    # if that word is also in lexicon 
                    index_val=lexicon.index(word.lower())
                    # we check index value of that word and make featureset 1 at that index 
                    features[index_val]=1
            features=list(features)
            featureset.append([features,classification])
            # featureset looks like [[[0,0,0,0,0,0,.....1],[0,1]],[[1,0,0,0,0,0,1....],[1,0]]]
    return featureset

def create_featureset_and_labels(pos,neg,test_size=0.1):
    lexicon=create_lexicon(pos,neg)
    features=[]
    features+=sample_handling('pos.txt',lexicon,[1,0])#like one hot coding 
    # first 'pos.txt' given as sample 
    features+=sample_handling('neg.txt', lexicon, [0,1])
    #second negative.txt' given as sample 
    random.shuffle(features)
    #features need to be shuffled 
    features=np.array(features)
    # converted to array 
    testing_size=int(test_size*len(features))
    train_x=list(features[:,0][:testing_size])# 90% data to train model features  
    train_y=list(features[:,1][:testing_size])#90% data to train model labels 
    test_x=list(features[:,0][-testing_size:])#last 10 % data to test model 
    test_y=list(features[:,1][-testing_size:])#last 10 % data to test model 
    return train_x,train_y,test_x,test_y
if __name__=='__main__':
    train_x,train_y,test_x,test_y=create_featureset_and_labels('pos.txt','neg.txt')
    