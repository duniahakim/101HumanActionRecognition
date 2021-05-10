from pathlib import Path
import cv2
import os


DATA_PATH = 'data/UCF50'
FRAME_DATA_PATH = 'data/UCF50_FRAME'


for dir in os.listdir(DATA_PATH):
    class_dir_path = os.path.join(DATA_PATH, dir)
    if not os.path.isdir(class_dir_path):
        continue

    for f in os.listdir(class_dir_path):
        video_path = os.path.join(class_dir_path, f)
        if not os.path.isfile(video_path):
            continue
        video_dir_name = f.replace('.avi', '')
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
