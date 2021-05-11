from numpy import asarray
from PIL import Image
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dense, Dropout,Flatten,LSTM
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


CHECKPOINT_PATH = "checkpoints/features_2epochs"


def get_CNN_model():
    model = Sequential()
    model.add(Conv2D(4, kernel_size=(3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(800, 2048, 1),padding ="same"))
    model.add(MaxPooling2D(pool_size=(1, 2), strides=(1, 2)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization(center=True, scale=True))
    model.add(Dropout(0.5))

    model.add(Conv2D(8, kernel_size=(2, 2), activation='relu', kernel_initializer='he_uniform'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
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
    # plot_model(model, to_file='plots/features.png', show_shapes=True, show_layer_names=True)

    return model


def main():
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=CHECKPOINT_PATH,
                        save_best_only=True, save_weights_only=False, verbose=1)
    history_checkpoint = LossHistory()
    labels = decompress_pickle('labels.pickle.pbz2')
    partition = decompress_pickle('partition.pickle.pbz2')
    training_generator = OurGenerator(partition['train'], labels, use_pretrained = True)
    validation_generator = OurGenerator(partition['val'], labels, use_pretrained = True)
    model = get_CNN_model()
    model.build()
    model.summary()
    history = model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    epochs=2,
                    callbacks=[cp_callback, history_checkpoint])
    model.save('models/features_2epochs')
    compressed_pickle('history/features_2epochs.pickle', history.history)
    plotLearningCurve(history)


if __name__ == "__main__":
    main()
