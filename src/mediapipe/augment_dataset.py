import vidaug.augmentors as va 
from PIL import Image, ImageSequence
import cv2
import numpy as np
import os
import random

from config import DATA_PATH, SPLITS

random.seed()

MAX_AUG = 5

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

def from_PIL_to_opencv(video_aug):
    cv2_frames=[]
    for frame in video_aug:
        open_cv_image = np.array(frame)
        cv2_frames.append(cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR))

    return cv2_frames


if __name__ == '__main__':

    dataset = f"top_10"
    to_aument = random.randint(1, MAX_AUG)

    for sp in SPLITS:
        for gloss in os.listdir(os.path.join(DATA_PATH, dataset, sp)):
            frames = []
            for video_name in os.listdir(os.path.join(DATA_PATH, dataset, sp, gloss)):
                video_path = os.path.join(DATA_PATH, dataset, sp, gloss, video_name)
                # get metadata from the video and encoding the output
                cap = cv2.VideoCapture(video_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                fourCC = cv2.VideoWriter_fourcc("m", "p", "4", "v")

                # retrieve frames as PIL images
                frames = video_loader(video_path)
                # get frame width and height
                frame_width, frame_height = frames[0].size

                for i in range(to_aument):
                    vid_name = video_name.split(".")[0]
                    aug_vid_name = f"{vid_name}_aug{i}.mp4"
                    aug_frames = data_transformer(frames, frame_height, frame_width, crop_factor=0.2)
                    new_frame_width, new_frame_height = aug_frames[0].size
                    cv2_frames = from_PIL_to_opencv(aug_frames)
                    path_out = os.path.join(DATA_PATH, dataset, sp, gloss, aug_vid_name)
                    out = cv2.VideoWriter(path_out, fourCC, fps, (new_frame_width, new_frame_height))

                    for frame in cv2_frames:
                        out.write(frame)

                    out.release()



