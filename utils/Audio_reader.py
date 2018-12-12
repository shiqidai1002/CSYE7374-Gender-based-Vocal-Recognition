# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 17:20:41 2018

@author: wenqi
"""
import tensorflow as tf
import numpy as np

'''
This function will take one datapoint and extract its embedding and label.
Return: embedding with shape (10,128) and label(s)
'''
def readsample(serialized_example):
    context_features = {
        'video_id': tf.FixedLenFeature([], dtype=tf.string),
        'start_time_seconds': tf.FixedLenFeature([], dtype=tf.float32),
        'end_time_seconds': tf.FixedLenFeature([], dtype=tf.float32),
        'labels': tf.VarLenFeature(dtype=tf.int64),
    }
    sequence_features = {
        'audio_embedding': tf.VarLenFeature(tf.string)
    }
    context, features = tf.parse_single_sequence_example(
        serialized_example, context_features, sequence_features)
    
    labels = context['labels'].values
    audio_embedding = features['audio_embedding'].values
#    video_id = context['video_id']
    with tf.Session() as sess:
        labels_r = sess.run(labels)
        audio_embedding_r = sess.run(audio_embedding)
#        video_id_r = sess.run(video_id)

    au_em=[]
    for i in range(len(audio_embedding_r)):
        au_em.append([int(x) for x in bytes(audio_embedding_r[i])]) 
    au_em=np.asarray(au_em)
    return au_em,labels_r  
    
    
'''
this function take a TFRecord which could contain mutilple datapoint. 
Return: two lists with embeddings and labels 
'''   
def readtf(file_path):
    tfrecord_file = file_path
    record_iterator = tf.python_io.tf_record_iterator(tfrecord_file)
    
    au_em_list=[]
    labels_list=[]
    
    for serialized_example in record_iterator:
        x,y=readsample(serialized_example)
        au_em_list.append(x)
        labels_list.append(y)
    return au_em_list,labels_list
    
