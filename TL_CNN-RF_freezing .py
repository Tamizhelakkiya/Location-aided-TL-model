#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 11:09:06 2022

@author: tamizh
"""
# Program to train the Keras ConvNet and use it for predictions


############ Import all the necessary modules #############

#from __future__ import division, print_function
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.models import Model,Sequential,load_model
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
import time
import pickle
import glob
from sklearn.tree import DecisionTreeClassifier

t1=time.time()
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



########### Define some 0f the functions #####################

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

train_path = '/media/tamizh/Backup Plus/CRL/CRL/DL_model_papersubmission_2.3.2021/highSNR/training_data'  # Folder for the data used for training
#print(os.listdir('/home/tamizh/Desktop/OFDM-CNN/training_data_OFDM_20'))
print(os.listdir('/media/tamizh/Backup Plus/CRL/CRL/DL_model_papersubmission_2.3.2021/highSNR/training_data'))
classes = [d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))] # Gets classes from all the folders mentioned under training_data folder
num_classes = len(classes)
data = dataset2journal.read_train_sets2(train_path, classes, validation_size=0.3) # Gets the data object using a class in dataset2.py


Xtrain = data.train.images
Ytrain = data.train.labels #one hot encodeing
Xtest = data.valid.images
Ytest = data.valid.labels 

model = load_model('/media/tamizh/Backup Plus/CRL/CRL/DL_model_papersubmission_2.3.2021/models/highSNR_Model_CNN_256')
model.summary()

for layer in model.layers:
	layer.trainable = False

new_model = Model(inputs=model.input, outputs=model.get_layer('flatten_1').output)
new_model.summary()

##RANDOM FOREST
RF_model = RandomForestClassifier(n_estimators=50, random_state=42)
x_for_RF=new_model.predict(Xtrain)
##
TL_model=RF_model.fit(x_for_RF,Ytrain)
pickle.dump(TL_model, open('model_CNN_TL_RF.sav', 'wb'))

print('CNN_RF_with_freezing model generated')
t2=time.time()
print('Model_gen time:',t2-t1)


def plot_cm(test_path,plot_extn,classes,model):
    data = read_train_sets2(test_path, classes, validation_size=0) # Gets the data object using a class in dataset2.py
    # num_classes = len(labels)
#    test_path = '/media/prabhuchandhar/E/CNN/testing_data_1'
    Xtest1 = data.train.images # Testing samples
    Ytest1 = data.train.labels # Testing labels
    print("Testing data prep done")
    
    
    x_test_feature = new_model.predict(Xtest1)   

    Ypred1 = TL_model.predict(x_test_feature) # Use model to predict
    # Ypred1 is in SoftMax form, convert to One-Hot Encoded
    index = np.argmax(Ypred1,axis=1) # Get index of maximum element of each testing example
    for i in range(Ypred1.shape[0]):
        for j in range(Ypred1.shape[1]):
            if j == index[i]:
                Ypred1[i][j] = 1
            else:
                Ypred1[i][j] = 0
    
    y_test = pd.Series(np.argmax(Ytest1,axis=1),name = 'Actual')    # Form dataframe of actual labels    
    y_pred = pd.Series(np.argmax(Ypred1,axis=1),name = 'Predicted') # Form dataframe of predicted labels
    
    df_confusion = pd.crosstab(y_test, y_pred) # Crosstab to get in required format
    df_conf_norm = df_confusion / df_confusion.sum(axis=1) # Normalise it
    sn.heatmap(df_conf_norm, annot=True, annot_kws={"size": 8}, cmap="Greens", linewidths=0.25, linecolor='black')
   
    s = df_confusion.to_numpy()
#    print(df_confusion)
    n = min(s.shape[0],s.shape[1])
    class_count = np.zeros(num_classes,dtype=int)
    for i in np.arange(n):
        class_count[i] = s[i,i]
    #plt3.matshow(df_conf_norm, cmap=plt3.cm.gray_r) # Plot figure
    #plt3.colorbar() # Colourbar
    #ax = plt3.gca() 
    #    plt3.locator_params(numticks=num_classes)
    plt3.xticks(np.arange(0,num_classes)+.5,classes,rotation='vertical',ha="center")
    plt3.yticks(np.arange(0,num_classes)+.5,classes,rotation='horizontal',va="center")
    plt3.xlim(0,num_classes+1)
    plt3.ylim(0,num_classes+1)
    plt3.ylabel(df_confusion.index.name)
    plt3.xlabel(df_confusion.columns.name)
    plt3.tight_layout()
#    plt3.show()
    plt3.savefig('/home/tamizh/Documents/CNN_Ieee_access/confusion_matrix_cnn_rf/'+plot_extn+'ConfusionMatrix_cnn_TL.png',dpi=200)
    plt3.close()
    plt3.clf()
    cm = confusion_matrix(y_test, y_pred)  
    return cm, classification_report(y_test, y_pred,output_dict=True),class_count

def test(train_scenario,test_scenario,bn,model,classes):
    plot_extn = train_scenario+'_'+str(int(test_scenario))+bn+'_'
    conf_matrix,classification_report,class_count = plot_cm(test_path,plot_extn,classes,model)
    print(classification_report)
    print(conf_matrix)
    return classification_report,class_count #,conf_matrix

y2 = np.zeros(97,dtype=float)
e1 = np.zeros(97,dtype=float)
e2 = np.zeros(97,dtype=float)
e3 = np.zeros(97,dtype=float)
e4 = np.zeros(97,dtype=float)
e5 = np.zeros(97,dtype=float)
e6 = np.zeros(97,dtype=float)
e7 = np.zeros(97,dtype=float)



model_name = 'CNN_RF_freezing'

train_scenario = 'high'
bn =''
for test_scenario in np.arange(0,97):
    print(int(test_scenario))
    if int(test_scenario)!=38 and int(test_scenario)!=39 and int(test_scenario)!=40 and int(test_scenario)!=80 and int(test_scenario)!=88 and int(test_scenario)!=89 and int(test_scenario)!=90 and int(test_scenario)!=91 and int(test_scenario)!=92 and int(test_scenario)!=93 and int(test_scenario)!=94 and int(test_scenario)!=95:
#        test_path = '/media/prabhuchandhar/E/CNN/highSNR/testing_data_'+str(int(test_scenario))  # Folder for the data used for testing
        test_path = '/media/tamizh/Backup Plus/CRL/CRL/DL_model_papersubmission_2.3.2021/CNN/testing_data_'+str(int(test_scenario)) 
#        x = test(train_scenario,str(int(test_scenario)),bn,model,classes)
        classes = ['16QAM','64QAM','BPSK','CPFSK','GFSK','GMSK','QPSK']
        print(test_path)
        x,class_count = test(train_scenario,str(int(test_scenario)),bn,TL_model,classes)
        y2[int(test_scenario)-1] = x['accuracy']
        e1[int(test_scenario)-1] = class_count[0]
        e2[int(test_scenario)-1] = class_count[1]
        e3[int(test_scenario)-1] = class_count[2]
        e4[int(test_scenario)-1] = class_count[3]
        e5[int(test_scenario)-1] = class_count[4]
        e6[int(test_scenario)-1] = class_count[5]
        e7[int(test_scenario)-1] = class_count[6]
        

e1=(e1/500)*100
e2=(e2/500)*100    
e3=(e3/500)*100    
e4=(e4/500)*100
e5=(e5/500)*100
e6=(e6/500)*100
e7=(e7/500)*100    
y2=y2*100
df = pd.DataFrame({classes[0] : e1, classes[1] : e2, classes[2] : e3, classes[3] : e4, classes[4] : e5, classes[5] : e6, classes[6] : e7, "Avg" : y2})
df.to_csv(model_name+'_data.csv', index=False)

t2=time.time()
#print(t2-t1)
