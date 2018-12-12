# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 00:33:13 2018

@author: wenqi
"""
import pandas as pd
import numpy as np
def merge_and(a,b):
    for i in range(len(a)):
        a[i]=a[i] and b[i]
    return a

def merge_or(a,b):
    for i in range(len(a)):
        bb = False if np.isnan(b[i]) else b[i]
        a[i]=a[i] or bb
    return a

def merge_not(a):
    for i in range(len(a)):
        a[i]=True if np.isnan(a[i]) else not a[i] 
    return a


def getmanlist(file_list,labels):
    male='/m/05zppz'
    female='/m/02zsn'
    child='/m/0ytgt'
    animal='/m/0jbk'
    use=[male,female,child]
    not_use=[animal]
    arr=[False]*labels.shape[0]
    print("Building...")
    for lab in use:
        for i in range(1,13):
            arr=merge_or(arr,labels['label_'+str(i)].str.contains(lab))
            
            
    print("Excluding...")       
    not_arr=[True]*labels.shape[0]
    for lab in not_use:
        for i in range(1,13):
            tem=merge_not(labels['label_'+str(i)].str.contains(lab))
            not_arr=merge_and(not_arr,tem)
    
    fin_arr = merge_and(arr,not_arr)
    useful = [s[:2] for s in labels[fin_arr]['YTID']]
    lis=[]
    print("Almost...")
    for s in useful:
        for file in file_list:
            if s == file[:2]:
                lis.append(file)
    print("Get list done...")
    return lis

    


        
    






