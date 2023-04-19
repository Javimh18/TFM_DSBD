
import cv2
import numpy as np
import os
import random
import cv2
import numpy as np
import vidaug.augmentors as va 
from PIL import Image
from tqdm import tqdm

from config import VIDEOS_PATH, SPLITS

def video_loader(path:str):
    frames = []
    cap = cv2.VideoCapture(path)
    ret = True
    while ret:
        ret,cv2_im = cap.read()
        if ret:
            converted = cv2.cvtColor(cv2_im,cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(converted))
                
    cap.release()
    return frames


def from_PIL_to_opencv(video_aug):
    cv2_frames=[]
    for frame in video_aug:
        open_cv_image = np.array(frame)
        cv2_frames.append(cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR))

    return cv2_frames


# https://github.com/okankop/vidaug
def data_transformer(video_frames, height_of_frame, width_of_frame, crop_factor=0.1):
    sometimes = lambda aug: va.Sometimes(0.5, aug) # Used to apply augmentor with 50% probability

    height_of_frame_trans = int((35*(height_of_frame + (height_of_frame/10)))/height_of_frame)
    width_of_frame_tras = int((35*(width_of_frame + (width_of_frame/5)))/width_of_frame)

    center_crop_height = int(height_of_frame * crop_factor)
    center_crop_width = int(width_of_frame * crop_factor)

    random_crop_height = int(height_of_frame * crop_factor)
    random_crop_width = int(width_of_frame * crop_factor)

    seq = va.Sequential([
        va.RandomRotate(degrees=5),
        va.CenterCrop(size=(height_of_frame-center_crop_height, 
                                     width_of_frame-center_crop_width)),
        va.RandomCrop(size=(height_of_frame-random_crop_height, 
                                      width_of_frame-random_crop_width)),
        va.RandomTranslate(x=width_of_frame_tras, y=height_of_frame_trans),
        va.HorizontalFlip(),
    ])

    return seq(video_frames)





    


