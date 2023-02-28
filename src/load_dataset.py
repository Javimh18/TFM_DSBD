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
from mp_funs import extract_landmarks_to_np
from six.moves import cPickle as pickle #for performance

EXTENSION = '.mp4'
SPLITS = ['train', 'val', 'test']

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities that will be useful for action representation

def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)

def load_dict(filename_):
    with open(filename_, 'rb') as f:
        ret_di = pickle.load(f)
    return ret_di

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

def load_transform_save_dataset(indexfile='data/WLASL_v0.3.json', vid_directory='data/videos', top_k = 200):
    content = json.load(open(indexfile))

    org_gloss_rank = organize(indexfile, vid_directory, top_k = 200)

    segmented_dataset = {}
    holistic_model_mp = mp_holistic.Holistic()

    for sp in SPLITS:
        segmented_dataset[sp] = []
        for gloss in org_gloss_rank.keys():
            for video_path in os.listdir(os.join.path(config.VIDEOS_PATH, sp, gloss)):
                cap = cv2.VideoCapture(video_path)

                # declaration of the holistic model in media pipe
                with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
                    video_as_landmarks = np.array([])
                    # Loop until the end of the video
                    while (cap.isOpened()):
                        # Capture frame by frame
                        ret, frame = cap.read()
                        if ret:
                            # Make detections
                            results = holistic.process(frame)
                            if results is None:
                                print("Something is wrong w/ mediapipe... Exiting.")
                                cap.release()
                                return
                            
                            # Once we get the landmarks, we insert them in a numpy array
                            # There are 4 landmarks that we are interested in: face-mesh pose, rigth-hand and left-hand 
                            # so we are going to extract the from the holistic process method

                            video_as_landmarks = np.append(video_as_landmarks, extract_landmarks_to_np(results))

                    cap.release()
                
            segmented_dataset[sp].append((video_as_landmarks, gloss))
    
    # with all correctly stored, we save the file in a pickl file.
    save_dict(segmented_dataset, 'data/npy_videos/npy_db.pkl')

    return segmented_dataset


if __name__ == '__main__':
    load_transform_save_dataset()
