from numpy import asarray
import keras
from PIL import Image
import numpy as np
from keras.models import Sequential
from keras.layers import Conv3D, MaxPooling3D, BatchNormalization, Dense, Dropout,Flatten,LSTM
from keras.losses import categorical_crossentropy
from keras.utils.vis_utils import plot_model
from keras.optimizers import Adam
import os
import random
from utils import compressed_pickle, decompress_pickle, plotLearningCurve
import pickle
import tensorflow as tf
from ourGenerator import OurGenerator
from tensorflow.keras import regularizers
from lossHistory import LossHistory


CHECKPOINT_PATH = "checkpoints/baseline_10epochs"

# print('is gpu available?')
# print(tf.config.list_physical_devices('GPU'))


def get_LSTM_model():
    model = Sequential()
    model.add(keras.Input(shape=(800, 2048)))
    model.add(LSTM(256))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(50, activation='softmax'))

    # Compile the model
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.001),
                  metrics=['accuracy'])

    # plot_model(model, to_file='images/baseline.png', show_shapes=True, show_layer_names=True)

    return model


def get_CNN_model():
    model = Sequential()
    model.add(Conv3D(32, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(10, 240, 320, 3),padding ="same"))
    model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(BatchNormalization(center=True, scale=True))
    model.add(Dropout(0.5))

    model.add(Conv3D(64, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(BatchNormalization(center=True, scale=True))
    model.add(Dropout(0.5))
    model.add(Flatten())

    model.add(Dense(256, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(101, activation='softmax'))

    # Compile the model
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.001),
                  metrics=['accuracy'])
    model.summary()
    # plot_model(model, to_file='images/baseline.png', show_shapes=True, show_layer_names=True)

    return model


def main():
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=CHECKPOINT_PATH,
                        save_best_only=True, save_weights_only=False, verbose=1)
    history_checkpoint = LossHistory()
    labels = decompress_pickle('labels.pickle.pbz2')
    partition = decompress_pickle('partition.pickle.pbz2')
    training_generator = OurGenerator(partition['train'], labels, use_pretrained = False)
    validation_generator = OurGenerator(partition['val'], labels, use_pretrained = False)
    model = get_CNN_model()
    model.build()
    model.summary()
    history = model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    epochs=10,
                    callbacks=[cp_callback, history_checkpoint])
    model.save('models/baseline_model')
    compressed_pickle('history/baseline_10epochs.pickle', history.history)
    # plotLearningCurve(history)


if __name__ == "__main__":
    main()
