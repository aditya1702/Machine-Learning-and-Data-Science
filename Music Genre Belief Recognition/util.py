#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 23:12:12 2018

@author: Vedant Choudhary and Aditya Vyas
@affiliation: Rutgers University, New Brunswick
"""

####################################################################
########################## util.py #################################
###  util.py is a utility program that stores some global        ###
###  variables and functions used quite often in other programs  ###
####################################################################
####################################################################

# Importing the required libraries for operations in the code
import numpy as np
import librosa as lbr
import matplotlib.pyplot as plt
import os

# Global variables which link to directory paths.
MAIN_DIR = os.path.dirname(os.path.abspath(__file__))
METADATA_DIR = MAIN_DIR + "/Data/fma_metadata/"
AUDIO_DIR = MAIN_DIR + "/Data/fma_small/"
DATA_DIR = MAIN_DIR + "/Data/"
PICKLE_DIR = MAIN_DIR + "/PickleData/"
MODEL_DIR = MAIN_DIR + "/Models/"
JS_STATIC_DIR = MAIN_DIR + "/static/model"
# Genres which the program will try to recognize. These are the genres present in fma_small dataset
GENRES = ['Electronic', 'Experimental', 'Folk', 'Hip-Hop', 'Instrumental',
          'International', 'Pop', 'Rock']

# Some hard-coded variables, which can be adjusted as per requirement
TRACK_COUNT = 8000
WINDOW_SIZE = 2048
WINDOW_STRIDE = WINDOW_SIZE // 2
N_MELS = 128
MEL_KWARGS = {
    'n_fft': WINDOW_SIZE,
    'hop_length': WINDOW_STRIDE,
    'n_mels': N_MELS
}

# Input - Filename and forceShape (to bring uniformity to the spectrograms)
# Return - A numpy array denoting values of the spectrogram
def load_track(filename, forceShape=None):
    sample_input, sample_rate = lbr.load(filename, mono=True)
    features = lbr.feature.melspectrogram(sample_input, **MEL_KWARGS).T
#    print(features.shape)
    if forceShape is not None:
        if features.shape[0] < forceShape[0]:
            delta_shape = (forceShape[0] - features.shape[0], forceShape[1])
            features = np.append(features, np.zeros(delta_shape), axis=0)
        elif features.shape[0] > forceShape[0]:
            features = features[: forceShape[0], :]

    features[features == 0] = 1e-6

    return (np.log(features), float(sample_input.shape[0]) / sample_rate)

if __name__ == "__main__":
    # USING LIBROSA EXAMPLE
    y, sr = load_track(AUDIO_DIR + "003/003270.mp3")
    plt.figure(figsize=(10, 4))
    lbr.display.specshow(y, y_axis='mel', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel spectrogram')
    plt.tight_layout()
