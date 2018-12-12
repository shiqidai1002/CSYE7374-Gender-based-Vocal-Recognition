# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 02:27:26 2018

@author: wenqi
"""

import os
from Audio_reader import readtf
from getman import getmanlist
import gc
import numpy as np
import pandas as pd
import pickle
from keras.layers import Dense, Input, Dropout,Flatten
from keras.layers import Conv1D, MaxPooling2D,BatchNormalization,Conv2D
from keras.models import Model
from keras.models import Sequential
from keras import optimizers
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.externals import joblib

model_1 = load_model('MLP_75.model')
model_2 = load_model('Conv1D.model')

X = []
Y = [] 
with open("X_list_eval.pickle","rb") as xf:
    X.extend(pickle.load(xf))
with open("Y_list_eval.pickle","rb") as xf:
    Y.extend(pickle.load(xf))
      
with open("X_list.pickle","rb") as xf:
    X.extend(pickle.load(xf))
with open("Y_list.pickle","rb") as xf:
    Y.extend(pickle.load(xf))

X=np.array(X)
Y=np.array(Y)  
X=np.reshape(X,(X.shape[0],X.shape[1],X.shape[2]))


X_1 = X.astype('float32')    
X_1 /= 255
y_pred_1 = model_1.predict(X_1)



X_2 = X.astype('float32')    
X_2 /= 255
y_pred_2 = model_2.predict(X_2)




x_ens=np.hstack((y_pred_1,y_pred_2))

def maxpos(li):
      if (li[1]>li[0]) and (li[1]>li[2])  :
        return 1
      if (li[2]>li[0]) and (li[2]>li[1])  :
        return 2     
      return 0

y_ens=[maxpos(y) for y in Y]

x_train, x_test, y_train, y_test = train_test_split(x_ens, y_ens, test_size=0.2, random_state=42)
dt_cla = DecisionTreeClassifier()
dt_cla.fit(x_train,y_train)
y_pred=dt_cla.predict(x_test)
print(confusion_matrix(y_true=y_test,y_pred=y_pred))
print(accuracy_score(y_test,y_pred))

joblib.dump(dt_cla,'ensemble_decisionTree_2.model')
