# coding: utf-8


# create a class : mnist
# normalize the data from kaggle official


import pandas as pd
from random import *
import numpy as np
class mnist:
    def __init__(self):
        self.tmp = 0
	self.test_tmp = 0

# default is train
    def read_csv_train(self,file):
        data = pd.read_csv(file)
        self.values =  data.values[:,1:785]
        self.labels =  data.values[:,0]
        self.labels = self.vectorization_label()

# with self.test indicate test set
    def read_csv_test(self,file):
        data = pd.read_csv(file)
        self.test_values =  data.values

    def vectorization_label(self):
        tmp = np.zeros([len(self.labels),10])
        for i in xrange(len(self.labels)):
            tmp[i][self.labels[i]] = 1
        return tmp

    def vectorization_label_test(self):
        tmp = np.zeros([len(self.test_labels),10])
        for i in xrange(len(self.test_labels)):
            tmp[i][self.test_labels[i]] = 1
        return tmp

    def read_next_batch(self,num):
        out = [self.values[self.tmp:self.tmp + num,:],self.labels[self.tmp
            :self.tmp + num,:]]

        self.tmp += num
	if self.tmp == len(self.values):
	    self.tmp = 0
        return out

    def test_next(self,num):
	out = self.test_values[self.test_tmp:self.test_tmp + num]
	self.test_tmp += num
	return out
    



