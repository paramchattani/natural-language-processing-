# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 10:31:56 2020

@author: param
"""

import tensorflow as tf 
from nltk_data import create_featureset_and_labels
import numpy as np
#method imported from file nltk_data
 
train_x,train_y,test_x,test_y=create_featureset_and_labels('pos.txt', 'neg.txt')

node_1=500
node_2=500
n_classes=2 # [0,1] & [1,0]
batch_size=128
hm_epochs=10
x=tf.placeholder('float',[None,len(train_x[0])])
y=tf.placeholder('float')# label for the data
def nn_model(data):
    hidden_layer_1={'weights':tf.Variable(tf.random_normal([len(train_x[0]),node_1])),'biases':tf.Variable(tf.random_normal([node_1]))}
    # we are deciding number of weighs and biases in each layer
    # i/p data * weights + bias
    hidden_layer_2={'weights':tf.Variable(tf.random_normal([node_1,node_2])),'biases':tf.Variable(tf.random_normal([node_2]))}
    output_layer={'weights':tf.Variable(tf.random_normal([node_2,n_classes])),'biases':tf.Variable(tf.random_normal([n_classes]))}
    l1=tf.add(tf.matmul(data,hidden_layer_1['weights']),hidden_layer_1['biases'])
    l1=tf.nn.relu(l1)
    l2=tf.add(tf.matmul(l1,hidden_layer_2['weights']),hidden_layer_2['biases'])
    l2=tf.nn.relu(l2)
    l3=tf.add(tf.matmul(l2,output_layer['weights']),output_layer['biases'])
    return l3

def train_neural_network(x):
    prediction=nn_model(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(hm_epochs):
            epoch_loss=0
            i=0
            while i<len(train_x):
                start=i
                end=i+batch_size
                ex=np.array(train_x[start:end])
                ey=np.array(train_y[start:end])
                _,c=sess.run([optimizer,cost],feed_dict={x:ex,y:ey})
                epoch_loss+=c
                i+=batch_size
            print('epochs',epoch+1,'completed out of ',hm_epochs,'epoch error',epoch_loss)
        correct=tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
        accuracy=tf.reduce_mean(tf.cast(correct,'float'))
        print('accuracy:',accuracy.eval({x:test_x,y:test_y}))
train_neural_network(x) 