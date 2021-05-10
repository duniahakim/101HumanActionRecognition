from numpy import asarray
import math
from PIL import Image
import numpy as np
from keras.models import Sequential
from keras.layers import Conv3D, MaxPooling3D, BatchNormalization, Dense, Dropout,Flatten
from keras.losses import categorical_crossentropy
from keras.utils.vis_utils import plot_model
from keras.optimizers import Adam
import os
import random
from utils import compressed_pickle, decompress_pickle, plotLearningCurve
import pickle
from keras.utils.np_utils import to_categorical


FRAME_DATA_PATH = 'data/UCF101_Frames'
FRAMES_PER_VIDEO = 10


def class_to_int():
    class_to_int = {}
    class_count = 0
    for class_dir in os.listdir(FRAME_DATA_PATH):
        class_dir_path = os.path.join(FRAME_DATA_PATH, class_dir)
        if not os.path.isdir(class_dir_path):
            continue
        class_to_int[class_dir] = class_count
        class_count += 1
    compressed_pickle('class_to_int.pickle', class_to_int)

def setupInput():
    class_to_int = decompress_pickle('class_to_int.pickle.pbz2')
    labels = {}
    ids = []
    counter = 0
    for class_dir in os.listdir(FRAME_DATA_PATH):
        class_dir_path = os.path.join(FRAME_DATA_PATH, class_dir)
        if not os.path.isdir(class_dir_path):
            continue
        label = class_to_int[class_dir]
        for video_dir in os.listdir(class_dir_path):
            video_dir_path = os.path.join(class_dir_path, video_dir)
            if not os.path.isdir(video_dir_path):
                continue
            frames = []
            all_frames =  os.listdir(video_dir_path)
            num_frames_choose = FRAMES_PER_VIDEO
            if len(all_frames) < FRAMES_PER_VIDEO:
                num_frames_choose = len(all_frames)
            chosen_frames_indices = np.linspace(0, len(all_frames) - 1, num = num_frames_choose)
            for frame_index in chosen_frames_indices:
                frame_index = math.floor(frame_index)
                frame_file = all_frames[frame_index]
                frame_path = os.path.join(video_dir_path, frame_file)
                if not os.path.isfile(frame_path):
                    continue
                image = Image.open(frame_path)
                image = normalize(image)
                frames.append(image)

            id = 'id-' + str(counter)
            ids.append(id)
            compressed_pickle('input/' + id + '.pickle', frames)
            counter += 1
            labels[id] = label

        print('saved inputs for ' + class_dir)


    training_ids = random.choices(ids, k = int(np.floor(len(ids) * 2 / 3)))
    rest = list(set(ids) - set(training_ids))
    val_ids = random.choices(rest, k = int(np.floor(len(rest) * 2 / 3)))
    test_ids = list(set(rest) - set(val_ids))
    partition =  {'train': training_ids, 'val': val_ids, 'test': test_ids}
    compressed_pickle('partition.pickle', partition)
    compressed_pickle('labels.pickle', labels)


def normalize(image):
    pixels = asarray(image)
    image.close()
    pixels = pixels.astype('float32')
    mean = pixels.mean()
    pixels = pixels - mean
    return pixels


def main():
    class_to_int()
    setupInput()


if __name__ == "__main__":
    main()
