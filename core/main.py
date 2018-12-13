# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 22:55:58 2018

@author: Sean
"""
import numpy as np
from scipy.io import wavfile
import six
import tensorflow as tf

import vggish_input
import vggish_params
import vggish_postprocess
import vggish_slim

import preprocessing

import keras

from sklearn.externals import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score

flags = tf.app.flags

flags.DEFINE_string(
    'checkpoint', 'vggish_model.ckpt',
    'Path to the VGGish checkpoint file.')

flags.DEFINE_string(
    'pca_params', 'vggish_pca_params.npz',
    'Path to the VGGish PCA parameters file.')

flags.DEFINE_string(
    'tfrecord_file', '../testing-the-bayou',
    'Path to a TFRecord file where embeddings will be written.')

FLAGS = flags.FLAGS

def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()    
    keys_list = [keys for keys in flags_dict]    
    for keys in keys_list:
        FLAGS.__delattr__(keys)


def embed(wavform_slice, rate):  
  norm_wavform_slice = preprocessing.normalize(wavform_slice)
  examples_batch = vggish_input.waveform_to_examples(norm_wavform_slice,rate)
  #print('examples_batch:')
  #print(examples_batch)
  print('examples_batch len: ' + str(len(examples_batch)))

  # Prepare a postprocessor to munge the model embeddings.
  pproc = vggish_postprocess.Postprocessor(FLAGS.pca_params)

  with tf.Graph().as_default(), tf.Session() as sess:
    # Define the model in inference mode, load the checkpoint, and
    # locate input and output tensors.
    vggish_slim.define_vggish_slim(training=False)
    vggish_slim.load_vggish_slim_checkpoint(sess, FLAGS.checkpoint)
    features_tensor = sess.graph.get_tensor_by_name(
    vggish_params.INPUT_TENSOR_NAME)
    embedding_tensor = sess.graph.get_tensor_by_name(
    vggish_params.OUTPUT_TENSOR_NAME)
    
    # Run inference and postprocessing.
    [embedding_batch] = sess.run([embedding_tensor],
                     feed_dict={features_tensor: examples_batch})
    #print('embedding_batch: ')
    #print(embedding_batch)
    #print(embedding_batch.shape)
    postprocessed_batch = pproc.postprocess(embedding_batch)
    print('postprocessed_batch: ')
    print(postprocessed_batch)
    print(postprocessed_batch.shape)
  return postprocessed_batch

def main(_):
    model_path1 = '../MLP_75.model'
    model_path2 = '../Conv1D.model'
    model_path3 = '../ensemble_decisionTree_2.model'
    wav_file = '../data/000000__balloonhead__welcome-to-the-bayou.wav'
    sliced_windows, times, rate = preprocessing.load_and_sliced(wav_file)
    embeded_windows = []
    for wavform_slice in sliced_windows:
        embeded_windows.append(embed(wavform_slice, rate))
    
    embeded_windows = np.asarray(embeded_windows)
    
    #convert to float and normalize
    embeded_windows = embeded_windows.astype('float32')
    embeded_windows /= 255
    print('embeded_windows shape:')
    print(embeded_windows.shape)
    
    #############
    #Conv2D model
    model1 = keras.models.load_model(model_path1)
    print('Conv2D model loaded')
    
    print('input shape: ')
    print(embeded_windows.shape)
    #print(embeded_windows1[0])
    predictions1 = model1.predict(embeded_windows)
    print("Predictions 1 : ")
    print(predictions1)
    
    #############
    #Conv1D model
    model2 = keras.models.load_model(model_path2)
    print('Conv1D model loaded')
    print('input shape: ')
    print(embeded_windows.shape)
    predictions2 = model2.predict(embeded_windows) #no need to reshape,
    print("Predictions 2 : ")
    print(predictions2)
    
    
    ###############
    #Ensemble model
    model3 =joblib.load(model_path3)
    print('Ensemble model loaded')
    #hstack two predictions from base models
    x_ens=np.hstack((predictions1,predictions2))
    predictions3 = model3.predict(x_ens)
    print("Predictions 3: ")
    print(predictions3)
    
    m=len(predictions3)
    n=m+9
    pred= [[0]*3]* n
    pred=np.asarray(pred)   
    pp=[0]*n
    for i in range(m):
        for j in range(10):
            pred[i+j,predictions3[i]] += 1
    for i in range(n):
        if (pred[i][0]>pred[i][1])and(pred[i][0]>pred[i][2]):
            pp[i] = 0
        else:
            if (pred[i][1]>pred[i][2])and(pred[i][1]>pred[i][0]):
                pp[i] = 1
            else:
                if (pred[i][2]>pred[i][1])and(pred[i][2]>pred[i][0]):
                    pp[i] = 2
                else:
                    pp[i] =pp[i-1]
    print('Final Prediction: ')
    print(pp)
  
    del_all_flags(tf.flags.FLAGS)
if __name__ == "__main__":
  tf.app.run()
