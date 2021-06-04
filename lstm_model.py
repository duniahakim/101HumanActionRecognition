from numpy import asarray
from PIL import Image
import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Conv3D, MaxPooling3D, BatchNormalization, Dense, Dropout, Flatten, LSTM
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


CHECKPOINT_PATH = "checkpoints/lstm_2epochs"


def get_LSTM_model():
    model = Sequential()
    model.add(keras.Input(shape=(800, 2048)))
    model.add(LSTM(256))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(101, activation='softmax'))

    # Compile the model
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.001),
                  metrics=['accuracy'])

    # plot_model(model, to_file='images/baseline.png', show_shapes=True, show_layer_names=True)

    return model


def main():
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=CHECKPOINT_PATH,
                        save_best_only=True, save_weights_only=False, verbose=1)
    history_checkpoint = LossHistory()
    labels = decompress_pickle('labels.pickle.pbz2')
    partition = decompress_pickle('partition.pickle.pbz2')
    training_generator = OurGenerator(partition['train'], labels, use_pretrained = True)
    validation_generator = OurGenerator(partition['val'], labels, use_pretrained = True)
    model = get_LSTM_model()
    model.build()
    model.summary()
    history = model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    epochs=2,
                    callbacks=[cp_callback, history_checkpoint])
    model.save('models/lstm_model')
    compressed_pickle('history/lstm_2epochs.pickle', history.history)
    plotLearningCurve(history.history)


if __name__ == "__main__":
    main()
