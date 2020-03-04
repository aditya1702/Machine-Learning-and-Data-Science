# This file contains all the architectures we tried. To implement these, just change the model in train_model.py

# Model 1 - Idea from the write up at http://deepsound.io/music_genre_recognition.html
def trainModel1(data, model_path):
  X = data['X']
  y = data['y']
  (X_train, X_val, y_train, y_val) = train_test_split(X, y, test_size = 0.3,
                                                      random_state = SEED)

  print('Building model...')
  EPOCHS = 50

#   n_features = x_train.shape[2]
  n_features = X_train.shape[2]
  input_shape = (None, n_features)
  model_input = Input(input_shape, name = 'input')
  layer = model_input
  for i in range(N_LAYERS):
      # second convolutional layer names are used by extract_filters.py
      layer = Conv1D(
              filters=CONV_FILTER_COUNT,
              kernel_size=FILTER_LENGTH,
              name='convolution_' + str(i + 1)
          )(layer)
      layer = BatchNormalization(momentum=0.9)(layer)
      layer = Activation('relu')(layer)
      layer = MaxPooling1D(2)(layer)
#       layer = Dropout(0.5)(layer)

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

  print('Training...')
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

# Model 2 - Using LSTM before time distributed layer - very slow
def trainModel2(data, model_path):
    X = data['X']
    y = data['y']
    (X_train, X_val, y_train, y_val) = train_test_split(X, y, test_size = 0.3,
                                                      random_state=SEED)

    print('Building model...')

    n_features = numFeatures
    input_shape = (None, n_features)
    model_input = Input(input_shape, name='input')
    layer = model_input
    for i in range(N_LAYERS):
      # second convolutional layer names are used by extract_filters.py
      layer = Conv1D(
              filters=CONV_FILTER_COUNT,
              kernel_size=FILTER_LENGTH,
              name='convolution_' + str(i + 1)
          )(layer)
    #       layer = BatchNormalization(momentum=0.7)(layer)
      layer = Activation('relu')(layer)
      layer = MaxPooling1D(2)(layer)
      layer = Dropout(0.5)(layer)

    #   layer = Dropout(0.5)(layer)
    layer = LSTM(LSTM_COUNT, return_sequences=True)(layer)
    #   layer = Dropout(0.5)(layer)
    layer = TimeDistributed(Dense(len(GENRES)))(layer)
    layer = Activation('softmax', name='output_realtime')(layer)
    time_distributed_merge_layer = Lambda(
          function=lambda x: K.mean(x, axis=1),
          output_shape=lambda shape: (shape[0],) + shape[2:],
          name='output_merged'
      )
    layer = time_distributed_merge_layer(layer)
    model_output = layer
    model = Model(model_input, model_output)
    opt = RMSprop(lr=0.001)
    model.compile(
          loss='categorical_crossentropy',
          optimizer=opt,
          metrics=['accuracy']
      )

    print('Training...')
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

# Model 3 - Spotify model through http://benanne.github.io/2014/08/05/spotify-cnns.html


def train_model_spotify(data, model_path):
  x = data['X']
  y = data['y']
  (x_train, x_val, y_train, y_val) = train_test_split(x, y, test_size=0.3, random_state=SEED)

  print ('Building model...')

  input_shape = (x_train.shape[1], x_train.shape[2])
  print (input_shape)
  model_input = Input(shape=input_shape)
  layer = model_input
  for i in range(2):
      layer = Conv1D(filters=256, kernel_size=FILTER_LENGTH,strides=2)(layer)
      layer = Activation('relu')(layer)
      layer = MaxPooling1D(2)(layer)
  averagePool = GlobalAveragePooling1D()(layer)
  maxPool = GlobalMaxPooling1D()(layer)
  layer = concatenate([averagePool, maxPool])
  layer = Dropout(rate=0.5)(layer)
  layer = Dense(units=len(GENRES))(layer)
  model_output = Activation('softmax')(layer)
  model = Model(model_input, model_output)
  opt = Adam()
  model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
  model.fit(
      x_train, y_train, batch_size=BATCH_SIZE, epochs=80,
      validation_data=(x_val, y_val),verbose=1, callbacks=[
         ModelCheckpoint(
              model_path, save_best_only=True, monitor='val_acc', verbose=1
          ),
         ReduceLROnPlateau(
              monitor='val_acc', factor=0.5, patience=8, min_delta=0.01,
              verbose=1
          )
      ]
  )
  return model

# Model 4 - using weighted values instead of mean to classify a genre from its probabilities
def weightedCustom(x):
  weights=np.arange(77)
  weights=weights[:,np.newaxis]
  weightsKeras=K.variable(value=weights)
  values_tensor=x
  out=values_tensor*weights
  out=K.sum(out,axis=1)
  out=tf.divide(out,3003)#1+2+3+.....+77
  return out

def trainModel4(data, model_path):
  x = data['X']
  y = data['y']
  (x_train, x_val, y_train, y_val) = train_test_split(x, y, test_size=0.3,
          random_state=SEED)

  print('Building model...')

  n_features = x_train.shape[2]
  input_shape = (None, n_features)
  model_input = Input(input_shape, name='input')
  layer = model_input
  for i in range(N_LAYERS):
    # convolutional layer names are used by extract_filters.py
    layer = Conv1D(
            filters=CONV_FILTER_COUNT,
            kernel_size=FILTER_LENGTH,
            name='convolution_' + str(i + 1)
        )(layer)
    layer = Activation('relu')(layer)
    layer = MaxPooling1D(2)(layer)
    layer = Dropout(0.5)(layer)

#   layer = Dropout(0.5)(layer)
#   layer = LSTM(LSTM_COUNT, return_sequences=True)(layer)
#   layer = Dropout(0.5)(layer)
  layer = TimeDistributed(Dense(len(GENRES)))(layer)
  time_distributed_merge_layer = Lambda(
          function=lambda x: weightedCustom(x),
          output_shape=lambda shape: (shape[0],) + shape[2:],
          name='output_merged'
      )
  model_output = time_distributed_merge_layer(layer)
  layer = Activation('softmax', name='output_realtime')(layer)
  model = Model(model_input, model_output)
  opt = Adam(lr=0.001)
  model.compile(
          loss='categorical_crossentropy',
          optimizer=opt,
          metrics=['accuracy']
      )

  print('Training...')
  model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=70,
            validation_data=(x_val, y_val), verbose=1, callbacks=[
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
