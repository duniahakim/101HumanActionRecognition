from vit import create_vit_classifier
from numpy import asarray
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


CHECKPOINT_PATH = "checkpoints/our_model_2epochs"


def main():
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=CHECKPOINT_PATH,
                        save_best_only=True, save_weights_only=False, verbose=1)
    history_checkpoint = LossHistory()
    labels = decompress_pickle('labels.pickle.pbz2')
    partition = decompress_pickle('partition.pickle.pbz2')
    training_generator = OurGenerator(partition['train'], labels, use_pretrained = False)
    validation_generator = OurGenerator(partition['val'], labels, use_pretrained = False)
    model = create_vit_classifier()
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.001),
                  metrics=['accuracy'])
    model.build(input_shape = (240, 3200, 3))
    model.summary()
    history = model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    epochs=2,
                    callbacks=[cp_callback, history_checkpoint])
    model.save('models/baseline_model')
    compressed_pickle('history/baseline_2epochs.pickle', history.history)
    plotLearningCurve(history)


if __name__ == "__main__":
    main()
