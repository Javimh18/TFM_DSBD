import json
import tensorflow as tf
import numpy as np
import cv2
import config_model
import os
import mediapipe as mp
import shutil
import time
import sys
import pandas as pd

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities that will be useful for action representation

import logging
logging.basicConfig(filename='download_{}.log'.format(int(time.time())), filemode='w', level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

EXTENSIONS = ['.mkv','.mp4'] # there may be more extensions, so a list is appropiate.

def mp_detect_pose(image, model):
    results = model.process(image)
    return image, results

def create_dataset():
    content = json.load(open(config_model.METADATA_PATH))

    frames = []
    labels = []
    holistic_model_mp = mp_holistic.Holistic()
    for entry in content:
        gloss = entry['gloss']
        instances = list(entry['instances'])
        
        for inst in instances:
            vid_id = inst['video_id']
            split = inst['split']
            cap = cv2.VideoCapture(os.join.path(config_model.VIDEOS_PATH, vid_id+".mp4"))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # once we've got the fps and the total_frames of the video
            # we create a numpy array and iterate in it with the frame rate defined
            frame_idx = np.arange(0, total_frames, fps)
            
            for idx in frame_idx:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    continue

                # Now we pass it to mp_detect_pose, that captures the left and right hand posture plus the pose
                image, results = mp_detect_pose(frame, holistic_model_mp)

def organize(indexfile='nil', vid_directory='videos'):
    if indexfile == 'nil':
        logging.info('No index specified. Exiting.')
        return

    content = json.load(open(indexfile))

    for entry in content:
        gloss = entry['gloss']
        instances = list(entry['instances'])
        
        for inst in instances:
            vid_id = inst['video_id']
            split = inst['split']

            if os.path.exists(os.path.join(vid_directory, split, gloss)): 
                os.mkdir(os.path.join(vid_directory, split, gloss))

            for ext in EXTENSIONS and flag == False:
                source = os.path.join(vid_directory, vid_id+ext)
                if os.path.exists(source):
                    flag = True
                    
            destination = os.path.join(vid_directory, split, gloss)
            shutil.move(source, destination)

if __name__ == '__main__':
    logging.info('Start downloading non-youtube videos.')
    organize('WLASL_v0.3.json')



    

