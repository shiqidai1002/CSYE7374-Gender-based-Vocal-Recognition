# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 22:26:55 2018

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

'''
This function checks labels of single datapoint.
74: animal
1: male
2: female
3: child

Return: Only keep 1, 2, and 3 labels. will mark 'animal' and other labels as 0,
        which is not in our scale and will be abandoned.
'''
def check(y):
    if 74 in y:
        return 0 
    #male
    if 1 in y:
        return 1
    #female
    if 2 in y:
        return 2
    #child
    if 3 in y:
        return 3
    return 0
   
'''    
file_base='C:\\Users\\wenqi\\Desktop\\7374final\\features\\audioset_v1_embeddings\\bal_train'
labels=pd.read_csv('C:\\Users\\wenqi\\Desktop\\7374final\\balanced_train_segments.csv')
file_list=os.listdir(file_base)
file_list=getmanlist(file_list,labels)
n =len(file_list)
print(n)

i=0
for file in file_list:
    x_list,y_list = readtf(os.path.join(file_base,file))
    for j in range(len(x_list)):
        x=x_list[j]
        y=y_list[j]
        y=check(y)
        if (y != 0) and (x.shape == (10,128)):
            print(file+' join')
            X.append(x)
            Y.append(y)
    gc.collect
    i+=1
    if i%10 == 0:
        print (i/n)
'''
'''
file_base='C:\\Users\\wenqi\\Desktop\\7374final\\features\\audioset_v1_embeddings\\eval'
labels=pd.read_csv('C:\\Users\\wenqi\\Desktop\\7374final\\eval_segments.csv')
labels[['label_1','label_2','label_3','label_4'
        ,'label_5','label_6','label_7','label_8'
        ,'label_9','label_10','label_11','label_12']]=labels[['label_1','label_2','label_3','label_4'
        ,'label_5','label_6','label_7','label_8'
        ,'label_9','label_10','label_11','label_12']].astype('str')
file_list=os.listdir(file_base)
file_list=getmanlist(file_list,labels)
n =len(file_list)
print(n)
i=0
for file in file_list:
    x_list,y_list = readtf(os.path.join(file_base,file))
    for j in range(len(x_list)):
        x=x_list[j]
        y=y_list[j]
        y=check(y)
        if (y != 0) and (x.shape == (10,128)):
            print(file+' join')
            X.append(x)
            Y.append(y)
    gc.collect
    i+=1
    if i%10 == 0:
        print (i/n) 
        gc.collect
'''

'''
X=np.asarray(X)
X=X.reshape(X.shape[0],X.shape[1],X.shape[2],1)
Y=to_categorical(Y)
Y=np.asarray([y[1:] for y in Y])

'''




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

X = X.astype('float32')    
X /= 255

learning_rate=0.001
decay=0.00001
momentum=0.9


model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',activation='relu',input_shape=X.shape[1:]))
model.add(BatchNormalization())
model.add(Conv2D(32, (3, 3), padding='same',activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(10,activation='relu'))
model.add(Dense(3,activation='softmax'))
model.summary()

optimizer = optimizers.SGD(lr=learning_rate,momentum=momentum,decay=decay)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer, metrics=['accuracy'])


history=model.fit(X, Y,
          batch_size=32,
          epochs=10,
          verbose=2,
          validation_split=0.3)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()














'''
X=[x for x in X]
size_x = [x.shape for x in X]
corr_x = [shape==(10,128) for shape in size_x]
fix_X = X[corr_x]
'''


'''
import pickle

with open("X_list_eval.pickle","wb") as xf:
    pickle.dump(X,xf)
    

with open("Y_list_eval.pickle","wb") as yf:
    pickle.dump(Y,yf)
'''
    