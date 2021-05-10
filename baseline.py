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
from input_prep import get_all_input, get_input_from_file
import tensorflow as tf
from ourGenerator import OurGenerator
from tensorflow.keras import regularizers
from lossHistory import LossHistory


CHECKPOINT_PATH = "checkpoints/ours_7_10epochs"

print('is gpu available?')
print(tf.config.list_physical_devices('GPU'))


def get_model():
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


def main():
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=CHECKPOINT_PATH,
                        save_best_only=True, save_weights_only=False, verbose=1)
    history_checkpoint = LossHistory()
    labels = decompress_pickle('labels.pickle.pbz2')
    partition = decompress_pickle('partition.pickle.pbz2')
    training_generator = OurGenerator(partition['train'], labels, use_pretrained = True)
    validation_generator = OurGenerator(partition['val'], labels, use_pretrained = True)
    model = get_model()
    model.build()
    model.summary()
    history = model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    epochs=10,
                    callbacks=[cp_callback, history_checkpoint])
    model.save('models/ours_7_10epochs_model')
    compressed_pickle('history/ours_7_10epochs.pickle', history.history)
    # plotLearningCurve(history)


if __name__ == "__main__":
    main()
