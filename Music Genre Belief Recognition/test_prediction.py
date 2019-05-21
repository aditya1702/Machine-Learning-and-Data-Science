#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 17:22:16 2018

@author: Vedant Choudhary and Aditya Vyas
@affiliation: Rutgers University, New Brunswick
"""

####################################################################
###################### test_prediction.py ##########################
###  test_prediction.py is kind of a utility code written just   ###
###  to check the prediction values of any song.                 ###
####################################################################
####################################################################

# Importing the required libraries for operations in the code
from tensorflow.python.keras.models import load_model
from util import AUDIO_DIR, MODEL_DIR, load_track
from create_pickle_data import getDefaultShape

# Loads the model to predict
def prediction(model_path):
    model = load_model(model_path)
    return model

if __name__ == "__main__":
    model_path = MODEL_DIR + 'model_cnn50_relu.h5'
    defaultShape = getDefaultShape()
    test_x, _ = load_track(AUDIO_DIR + "003/003270.mp3")
    model = prediction(model_path)
    model.predict(test_x)
