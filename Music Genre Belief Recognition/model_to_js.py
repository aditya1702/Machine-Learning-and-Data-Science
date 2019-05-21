#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Dec 6 15:22:16 2018

@author: Vedant Choudhary and Aditya Vyas
@affiliation: Rutgers University, New Brunswick
"""

####################################################################
######################## model_to_js.py ############################
###  test_prediction.py is kind of a utility code written just   ###
###  to check the prediction values of any song.                 ###
####################################################################
####################################################################

# Importing the required libraries for operations in the code
from tensorflow.python.keras.models import Model, load_model
import tensorflowjs as tfjs
from util import MODEL_DIR, JS_STATIC_DIR

# Input - model in .h5 format
# Return -  model with input as input layer and output as output_realtime layer from the
#           architecture
def extract_realtime_model(full_model):
    input = full_model.get_layer('input').input
    output = full_model.get_layer('output_realtime').output
    model = Model(inputs=input, outputs=output)
    return model

# Input - model path, output path: where the model in json will be saved (to be used for web)
# Return - None
def convert_to_js(model_path, output_path):
    model = load_model(model_path)
    realtime_model = extract_realtime_model(model)
    realtime_model.compile(optimizer=model.optimizer, loss=model.loss)
    tfjs.converters.save_keras_model(realtime_model, output_path)

if __name__ == "__main__":
    model_path = MODEL_DIR + 'model_cnn50_relu.h5'
    output_path = JS_STATIC_DIR
    convert_to_js(model_path, output_path)
