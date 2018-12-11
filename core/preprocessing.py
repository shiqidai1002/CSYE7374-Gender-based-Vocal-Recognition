# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 22:20:38 2018

@author: Sean
"""
from scipy.io import wavfile

#file to separate
fname = ''

def windows(signal, window_size, step_size):
    if type(window_size) is not int:
        raise AttributeError("Window size must be an integer.")
    if type(step_size) is not int:
        raise AttributeError("Step size must be an integer.")
    windows = []
    times = []
    for i_start in range(0, len(signal), step_size):
        i_end = i_start + window_size
        if i_end >= len(signal):
            break
        times.append((i_start,i_end))
        windows.append(signal[i_start:i_end])
    return windows, times

'''
This function takes a file path of wave file which needed to be sliced.
Returns: sliced wavforms and list of tuple of start and end time.
'''
def load_and_sliced(fname):
    #get wavform
    rate, data = wavfile.read(fname)
    print("Sampling (frame) rate = ", rate)
    print("Total samples (frames) = ", data.shape)
    duration = len(data)/rate
    print("Duration = ", duration)
    if duration < 10:
        print("The length of input audio should be larger than 10s")
        return
    #prepare to slice
    window_duration = 10
    step_duration = window_duration / 10
    sample_rate = rate
    
    window_size = int(window_duration * sample_rate)
    step_size = int(step_duration * sample_rate)

    sliced_windows, times = windows(data, window_size, step_size)
    return sliced_windows, times, rate


def normalize(wav_data):
    samples = wav_data / 32768.0  # Convert to [-1.0, +1.0]
    return samples