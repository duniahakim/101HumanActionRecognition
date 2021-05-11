# This code is taken from a Stack Over Flow answer!

import json, codecs
import keras
import os


history_filename = 'history-checkpoints/features_2epochs.json'

def saveHist(path, history):
    with codecs.open(path, 'w+', encoding='utf-8') as f:
        json.dump(history, f, separators=(',', ':'), sort_keys=True, indent=4)

def loadHist(path):
    n = {}
    if os.path.exists(path):
        with codecs.open(path, 'r', encoding='utf-8') as f:
            n = json.loads(f.read())
    return n

def appendHist(h1, h2):
    if h1 == {}:
        return h2
    else:
        dest = {}
        for key, value in h1.items():
            dest[key] = value + h2[key]
        return dest

class LossHistory(keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs = None):
        new_history = {}
        for k, v in logs.items():
            new_history[k] = [v]
        current_history = loadHist(history_filename)
        current_history = appendHist(current_history, new_history)
        saveHist(history_filename, current_history)
