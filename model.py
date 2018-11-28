import random
import numpy as np
from keras.models import Sequential
from keras.layers import Conv1D, LSTM, Dense, MaxPooling1D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard

import keras.backend as K

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.75
set_session(tf.Session(config = config))

def build_model():
    model = Sequential()
    
    model.add(Conv1D(64, input_shape = (96, 21), kernel_size = 3, strides=1, padding='same', activation= 'relu'))
    model.add(Conv1D(64, kernel_size = 3, strides=1, padding='same', activation= 'relu'))
    model.add(MaxPooling1D(pool_size=2, stride=1, padding='same'))

    model.add(Conv1D(128, kernel_size = 3, strides=1, padding='same', activation= 'relu'))
    model.add(Conv1D(128, kernel_size = 3, strides=1, padding='same', activation= 'relu'))
    model.add(MaxPooling1D(pool_size=2, stride=1, padding='same'))
    
    model.add(Conv1D(256, kernel_size = 3, strides=1, padding='same', activation= 'relu'))
    model.add(Conv1D(256, kernel_size = 3, strides=1, padding='same', activation= 'relu'))
    model.add(MaxPooling1D(pool_size=2, stride=1, padding='same'))

    model.add(LSTM(64,  return_sequences = True))
    model.add(LSTM(64,  return_sequences = True))
    model.add(Dense(3,  activation='softmax'))
    model.summary()

    return model

def matthews_correlation(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos
        
    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos
    
    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)
          
    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)
                  
    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + K.epsilon())

if __name__ == "__main__":
    x= np.load('data/data_x.npy')
    y = np.load('data/data_y.npy')

    random_seed = 1126
    random.Random(random_seed).shuffle(x)
    random.Random(random_seed).shuffle(y)

    data_count = x.shape[0]
    modelname = 'c_64_128_256r_64_with_pooling'
    training_split = 0.6
    training_size = int(data_count * training_split)
    batch_size = 32
    epochs = 100

    training_x, validation_x = np.split(x, [training_size])
    training_y, validation_y = np.split(y, [training_size])

    model = build_model()
   
    model_ckpt = ModelCheckpoint('model/%s.h5' % modelname, verbose = 1, save_best_only = True)
    tensorboard = TensorBoard(log_dir='./logs/%s' % modelname, histogram_freq=0, write_graph=True, write_images=False)

    model.compile(loss = 'categorical_crossentropy', optimizer = Adam(lr = 1e-4),  metrics = [matthews_correlation])
    model.fit(
            x = training_x, y = training_y,
            validation_data = (validation_x, validation_y),
            batch_size= batch_size,
            epochs = epochs,
            callbacks = [model_ckpt, tensorboard])
