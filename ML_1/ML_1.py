#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 20:10:31 2023

@author: tonyyao
"""
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

#get data from .csv
train_set = pd.read_csv("./mnist_train.csv", header = None)
test_set = pd.read_csv("./mnist_test.csv", header = None) 


def raw_matrix(set, num1, num2):
    set_num1 = set[set[0] == num1]
    set_num2 = set[set[0] == num2]
    
    set_num1[0] = 0
    set_num2[0] = 1
    new_set = pd.concat([set_num1, set_num2])
    new_set.index = [i for i in range(len(new_set))]
    return new_set

#create raw variables matix
original_matrix = raw_matrix(train_set, 4, 6)

def rescale_matrix(set):
    y = set[[0]]
    x = set.loc[:, 1:] / 255
    return x, y

#create x, y variables matrix
x, y = rescale_matrix(original_matrix)

#initialize logistic regression
clf = LogisticRegression().fit(x, y)

#test classifier
test_original_matrix = raw_matrix(test_set, 4, 6)
x_test, y_test = rescale_matrix(test_original_matrix)
clf.score(x_test, y_test)

clf_nn = MLPClassifier(hidden_layer_sizes=(28),activation = 'logistic').fit(x, y)

