#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 17:22:16 2018

@author: Vedant Choudhary and Aditya Vyas
@affiliation: Rutgers University, New Brunswick
"""

####################################################################
######################## train_model.py ############################
###  train_model.py consists the model architecture for our      ###
###  program. Various other architectures were tried, but for    ###
###  clarity, only the best model is shown in the code. The      ###
###  documentation for our project will give a more expansive    ###
###  view of all the architectures tried                         ###
####################################################################
####################################################################

# Importing the required libraries for operations in the code
from util import GENRES, DATA_DIR, MODEL_DIR
from sklearn.model_selection import train_test_split
import pickle
from tensorflow.python.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Input, Conv1D, MaxPooling1D, Activation, Dropout, \
        BatchNormalization, Dense, Lambda, TimeDistributed

# Parameters for the model are specified as global variables
EPOCHS = 50
SEED = 42
N_LAYERS = 3
FILTER_LENGTH = 5
CONV_FILTER_COUNT = 256
BATCH_SIZE = 32
numFeatures = 128
#LSTM_COUNT = 256

# Using the model architecture to train the dataset
# Input - the consolidated pickle data, along with the path where model .h5 should be saved
def trainModel(data, model_path):
    X = data['X']
    y = data['y']
    (X_train, X_val, y_train, y_val) = train_test_split(X, y, test_size = 0.3, random_state=SEED)

    print('Building model using C-RNN architecture...')

    n_features = numFeatures
    input_shape = (None, n_features)
    model_input = Input(input_shape, name='input')
    layer = model_input
    for i in range(N_LAYERS):
        layer = Conv1D(
                filters=CONV_FILTER_COUNT,
                kernel_size=FILTER_LENGTH,
                name='convolution_' + str(i + 1)
                )(layer)
        layer = BatchNormalization(momentum=0.9)(layer)
        layer = Activation('relu')(layer)
        layer = MaxPooling1D(2)(layer)
        # layer = Dropout(0.5)(layer)

    layer = TimeDistributed(Dense(len(GENRES)))(layer)
    time_distributed_merge_layer = Lambda(
                function=lambda x: K.mean(x, axis=1),
                output_shape=lambda shape: (shape[0],) + shape[2:],
                name='output_merged'
                )
    layer = time_distributed_merge_layer(layer)
    layer = Activation('softmax', name='output_realtime')(layer)
    model_output = layer
    model = Model(model_input, model_output)
    opt = Adam(lr=0.001)
    model.compile(
            loss='categorical_crossentropy',
            optimizer=opt,
            metrics=['accuracy']
            )

    print('Starting training...')
    model.fit(
            X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS,
            validation_data=(X_val, y_val), verbose=1, callbacks=[
            ModelCheckpoint(
                    model_path, save_best_only=True, monitor='val_acc', verbose=1
                    ),
            ReduceLROnPlateau(
                    monitor='val_acc', factor=0.5, patience=10, min_delta=0.01,
                    verbose=1
                    )
            ]
            )
    return model

if __name__ == "__main__":
    with open(DATA_DIR + "finalé.pkl", "rb") as f:
        data = pickle.load(f)

    trainModel(data, MODEL_DIR + "model.h5")
