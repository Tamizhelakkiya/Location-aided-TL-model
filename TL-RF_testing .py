#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 07:08:43 2022

@author: tamizh
"""
# Program to train the Keras ConvNet and use it for predictions


############ Import all the necessary modules #############

#from __future__ import division, print_function
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.models import Model,Sequential
import numpy as np
import scipy.signal as signal
import time
import os, sys, argparse
from sklearn.utils import shuffle
import dataset2journal
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt3
import pandas as pd
import seaborn as sn
#from pandas_ml import ConfusionMatrix
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
#from sklearn.metrics import precision_score, recall_score,f1_score
#from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import classification_report
#from rtlsdr import RtlSdr
import pickle
import glob
from sklearn.tree import DecisionTreeClassifier


class DataSet2(object):

    def __init__(self, images, labels, img_names, cls):
        self._num_examples = images.shape[0]

        self._images = images
        self._labels = labels
        self._img_names = img_names
        self._cls = cls
        self._epochs_done = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def img_names(self):
        return self._img_names

    @property
    def cls(self):
        return self._cls

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_done(self):
        return self._epochs_done

    def next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size

        if self._index_in_epoch > self._num_examples:
            self._epochs_done += 1
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch

        return self._images[start:end], self._labels[start:end], self._img_names[start:end], self._cls[start:end]

    

def read_train_sets2(train_path, classes, validation_size):
    class DataSets(object):
        pass

    data_sets = DataSets()

    images, labels, img_names, cls = load_train2(train_path, classes)  # 2 calculating, 3 loading
    images, labels, img_names, cls = shuffle(images, labels, img_names, cls)

    if isinstance(validation_size, float):
        validation_size = int(validation_size * images.shape[0])

    validation_images = images[:validation_size]
    validation_labels = labels[:validation_size]
    validation_img_names = img_names[:validation_size]
    validation_cls = cls[:validation_size]

    train_images = images[validation_size:]
    train_labels = labels[validation_size:]
    train_img_names = img_names[validation_size:]
    train_cls = cls[validation_size:]

    data_sets.train = DataSet2(train_images, train_labels, train_img_names, train_cls)
    data_sets.valid = DataSet2(validation_images, validation_labels, validation_img_names, validation_cls)

    return data_sets


def load_train2(train_path, classes):
    samples = []
    labels = []
    sample_names = []
    cls = []

    for fields in classes:
        index = classes.index(fields)
        path = os.path.join(train_path, fields, '*.npy')
        files = glob.glob(path)
        
#        for fl in files[:75]:
        for fl in files:
            iq_samples = np.load(fl)
            real = np.real(iq_samples)
            imag = np.imag(iq_samples)

            iq_samples = np.ravel(np.column_stack((real, imag)))

            multiple = True
            if multiple:
                iq_samples1 = iq_samples[:1568]
                iq_samples1 = iq_samples1.reshape(28, 28, 2)
                iq_samples2 = iq_samples[1568:3136]
                iq_samples2 = iq_samples2.reshape(28, 28, 2)
                iq_samples3 = iq_samples[3136:4704]
                iq_samples3 = iq_samples3.reshape(28, 28, 2)
                iq_samples4 = iq_samples[4704:6272]
                iq_samples4 = iq_samples4.reshape(28, 28, 2)
                

                samples.append(iq_samples1)
                samples.append(iq_samples2)
                samples.append(iq_samples3)
                samples.append(iq_samples4)
#                 samples.append(iq_samples5)

                flbase = os.path.basename(fl)
                label = np.zeros(len(classes))
                label[index] = 1.0

                labels.append(label)
                labels.append(label)
                labels.append(label)
                labels.append(label)
#                 labels.append(label)
                sample_names.append(flbase)
                sample_names.append(flbase)
                sample_names.append(flbase)
                sample_names.append(flbase)
#                 sample_names.append(flbase)
                cls.append(fields)
                cls.append(fields)
                cls.append(fields)
                cls.append(fields)
#                 cls.append(fields)
                
            else:
                iq_samples = iq_samples[:6272]
                iq_samples = iq_samples.reshape(1, 6272, 1)
                samples.append(iq_samples)

                label = np.zeros(len(classes))
                label[index] = 1.0
                labels.append(label)
                flbase = os.path.basename(fl)
                sample_names.append(flbase)
                cls.append(fields)

    samples = np.array(samples)
    labels = np.array(labels)
    sample_names = np.array(sample_names)
    cls = np.array(cls)
    return samples, labels, sample_names, cls


########## Define some 0f the functions #####################

#Parser function that handles parsing of command line arguments

def build_parser():
    parser = argparse.ArgumentParser(description='Prepare the data')
    parser.add_argument('-decimation_rate', dest = 'decimation_rate', type = int, 
         default = 12, help = 'Decimation rate of the signal')
    parser.add_argument('-sampling_rate', dest = 'sampling_rate', type = int, 
         default = 2400000, help = 'Sampling rate of the signal')
    parser.add_argument('-sdr', dest = 'sdr', type = int, 
         default = 1, help = 'Read samples from file (0) or device (1)')
    parser.add_argument('-train_scenario', dest = 'train_scenario', type = str,
         default = 'high', help = 'Train scenario')
    parser.add_argument('-test_scenario', dest = 'test_scenario', type = int,
         default = 1, help = 'Test scenario)')
    return parser



def prepare_args():
    # hack, http://stackoverflow.com/questions/9025204/
    for i, arg in enumerate(sys.argv):
        if (arg[0] == '-') and arg[1].isdigit():
            sys.argv[i] = ' ' + arg
    parser = build_parser()
    args = parser.parse_args()
    return args


# Function to read the samples from _prediction_samples.dat files

def read_samples_sdr(freq):
    sdr = RtlSdr()
    sdr.sample_rate = sample_rate
    sdr.err_ppm = 56   # change it to yours
    sdr.gain = 'auto'

    f_offset = 250000  # shifted tune to avoid DC
    sdr.center_freq = freq - f_offset
    time.sleep(0.06)
    iq_samples = sdr.read_samples(1221376)
    iq_samples = iq_samples[0:600000]
    fc1 = np.exp(-1.0j * 2.0 * np.pi * f_offset / sample_rate * np.arange(len(iq_samples)))  # shift down 250kHz
    iq_samples = iq_samples * fc1
    return iq_samples


def read_samples_fc32(filename):
    iq_samples = np.fromfile(filename, np.complex64)# + np.int8(-127) #adding a signed int8 to an unsigned one results in an int16 array
    return iq_samples/64


def read_samples(filename):
#    f_offset = 250000  # Shifted tune to avoid DC
    samp = np.fromfile(filename,np.uint8)+np.int8(-127) # Adding a signed int8 to an unsigned one results in an int16 array
    x1 = samp[::2]/128 # Even samples are real(In-phase)
    x2 = samp[1::2]/128 # Odd samples are imaginary(Quadrature-phase)
    iq_samples = x1+x2*1j # Create the complex data samples
    iq_samples = iq_samples[0:600000]
    return iq_samples


scenario='high'

args = prepare_args()  # Get the decimation rate from command line
train_scenario = args.train_scenario #'low'
#test_scenario = args.test_scenario #'high'
bn=''

DIM1 = 28
DIM2 = 28
INPUT_DIM = 1568

  

test_path = '/media/tamizh/Backup Plus/CRL/CRL/DL_model_papersubmission_2.3.2021/CNN/highSNR/testing_data'  # Folder for the data used for testing
classes = [d for d in os.listdir(test_path) if os.path.isdir(os.path.join(test_path, d))] # Gets classes from all the folders mentioned under testing_data folder
#classes = ['16QAM','64QAM','BPSK','CPFSK','GFSK','GMSK','QPSK']
model = keras.models.load_model('highSNR_Model_rf')
#history = pickle.load(open('highSNR_history_rf','rb'))
#cnn_model.summary()
#model=pickle.load(open('model_TL_RF.sav', 'rb'))

from sklearn.metrics import accuracy_score

data = read_train_sets2(test_path, classes, validation_size=0) # Gets the data object using a class in dataset2.py
    
Xtest1 = data.train.images # Testing samples
Ytest1 = data.train.labels # Testing labels
print("Testing data prep done")

x_for_RF=feature_extractor.predict(Xtest1)


Ypred1 = model.predict(x_for_RF) 
print("tset.loss,test acc:",accuracy_score(Ypred1,Ytest1))

