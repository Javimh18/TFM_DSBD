import json
import os
import shutil
import itertools
import tensorflow as tf
import numpy as np
import cv2
import config
import os
import mediapipe as mp
import pandas as pd

EXTENSION = '.mp4'
SPLITS = ['train', 'val', 'test']
mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities that will be useful for action representation

def rm_error_info(func, path, _):
    print("INFO: The path", path, "does not exist. Skipping...")

def gloss_ranking(content, vid_directory, top_k):
    ranking = {}
    for entry in content:
        gloss = entry['gloss']
        
        appereances = []
        for split in SPLITS:
            path_to_check = os.path.join(vid_directory, split, gloss)
            items = os.listdir(path_to_check)
            appereances.append(len(items))

        # we add up all the videos from each split, given a gloss
        ranking[gloss] = sum(appereances)

    # sorting the dictionary based on the value
    top_ranking = {k: v for k, v in sorted(ranking.items(), key=lambda item: item[1], reverse=True)}

    return dict(itertools.islice(top_ranking.items(), top_k))


def organize(indexfile='data/WLASL_v0.3.json', vid_directory='data/videos', top_k = 200):
    if indexfile == 'nil':
        print('No index specified. Exiting.')
        return

    content = json.load(open(indexfile))

    for entry in content:
        gloss = entry['gloss']
        instances = entry['instances']
        
        for inst in instances:
            vid_id = inst['video_id']
            split = inst['split']

            source = os.path.join(vid_directory, vid_id+EXTENSION)

            # There might be some videos missing so, we check that they exist, and then
            # we put them in their corresponding folder
            if not os.path.exists(os.path.join(vid_directory, split, gloss)): 
                os.makedirs(os.path.join(vid_directory, split, gloss))

            if os.path.exists(source):
                destination = os.path.join(vid_directory, split, gloss)
                shutil.copy(source, destination)

    # We make a list of tuples, for the ranking of most glosses in the dataset
    # The tuple has the following structure (dir, split, gloss) and its already sorted
    gloss_rank = gloss_ranking(content, vid_directory, top_k)
    
    for entry in content:
        gloss = entry['gloss']
        instances = entry['instances']

        for sp in SPLITS:
            source_to_erase = os.path.join(vid_directory, sp, gloss)
            flag = False
            # If the gloss is found in the ranking, than we let it be, if not, we erase it
            for gl in gloss_rank.keys():
                source_in_ranking = os.path.join(vid_directory, sp, gl)
                if source_to_erase == source_in_ranking:
                    flag = True
            
            if flag == False:
                shutil.rmtree(source_to_erase, onerror=rm_error_info)

    return gloss_rank

def detect_pose(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    results = model.process(image)
    return image, results

def load_and_transform_dataset(indexfile='data/WLASL_v0.3.json', vid_directory='data/videos', top_k = 200):
    content = json.load(open(indexfile))

    org_gloss_rank = organize(indexfile, vid_directory, top_k = 200)

    frames = []
    labels = []
    holistic_model_mp = mp_holistic.Holistic()

    for sp in SPLITS:
        for gloss in org_gloss_rank.keys():
            for video in os.listdir(os.join.path(config.VIDEOS_PATH, sp, gloss)):
                cap = cv2.VideoCapture(os.join.path(config.VIDEOS_PATH, sp, gloss, video))
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
                    image, results = detect_pose(frame, holistic_model_mp)