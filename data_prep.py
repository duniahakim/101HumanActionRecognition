from pathlib import Path
import cv2
import os
import random


DATA_PATH = 'data/UCF101'
FRAME_DATA_PATH = 'data/UCF101_Frames'
NUM_VIDEOS_PER_CLASS = 50


for dir in os.listdir(DATA_PATH):
    class_dir_path = os.path.join(DATA_PATH, dir)
    if not os.path.isdir(class_dir_path):
        continue

    video_files = os.listdir(class_dir_path)
    random.shuffle(video_files)
    for i in range(NUM_VIDEOS_PER_CLASS):
        video_file = video_files[i]
        video_path = os.path.join(class_dir_path, video_file)
        if not os.path.isfile(video_path):
            continue
        video_dir_name = video_file.replace('.avi', '')
        video_dir_path = os.path.join(os.path.join(FRAME_DATA_PATH, dir), video_dir_name)
        if os.path.exists(video_dir_path):
            continue
        Path(video_dir_path).mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        count = 0
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == False:
                break
            image_name = video_dir_name + '_' + str(count) + '.jpg'
            cv2.imwrite(os.path.join(video_dir_path, image_name), frame)
            count += 1
        cap.release()
        cv2.destroyAllWindows()
