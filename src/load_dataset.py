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
import torch
from mp_funs import extract_landmarks_to_np, FACEMESH_LANDMARKS, POSE_LANDMARKS, HAND_LANDMARKS
from utils import save_dict, load_dict
from math import floor

EXTENSION = '.mp4'
SPLITS = ['train', 'val', 'test']
X_PICK_FILE_PATH = 'data/npy_videos/npy_db_x.pkl'
Y_PICK_FILE_PATH = 'data/npy_videos/npy_db_y.pkl'

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
            # If there is a gloss that does not appear in a split, we skip it
            try:
                items = os.listdir(path_to_check)
                appereances.append(len(items))
            except OSError:
                appereances.append(0)

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
    print("Ranking created with top", top_k, "glosses/labels...")
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


def from_dict_to_tensor(X):

    max_len = -10e8
    n_frame_lm = FACEMESH_LANDMARKS + POSE_LANDMARKS + 2*HAND_LANDMARKS

    for sp in SPLITS:
        split = X[sp]
        for video in split:
            cur_len = len(video)
            if cur_len > max_len:
                max_len = cur_len

    # once we got the max_len, we expland the videos to match the frames
    for sp in SPLITS:
        split = X[sp]
        for video in split:
            diff = len(video) - max_len
            if diff != 0:
                if diff % 2 != 0:
                    # insert at the end of the list
                    for i in range(0, floor(diff/2)):
                        video.append(np.zeros(n_frame_lm))
                    
                    # insert at the beginning of the list
                    for i in range(0, floor(diff/2)+1):
                        video.insert(i, np.zeros(n_frame_lm))
                else:
                    # insert at the end and the beginning of the list
                    for i in range(0, diff/2):
                        video.append(np.zeros(n_frame_lm))
                        video.insert(i, np.zeros(n_frame_lm))

    # Now that the # of frames between videos are matched, we cast them into tensors
    for sp in SPLITS:
        X[sp] = torch.tesor(X[sp])

    return X

def encode_labels(labels):
    pass 

def load_transform_save_dataset(indexfile='data/WLASL_v0.3.json', vid_directory='data/videos', top_k=200, save=False):

    # TODO: error handling in case the user specifies more top_k than glosses available 

    print("Organizing the videos...")
    org_gloss_rank = organize(indexfile, vid_directory, top_k)

    print("Folders arranged, applying mediapipe transformations to the dataset...")

    X = {}
    Y = {}
    # declaration of the holistic model in media pipe
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        # we iterate through the folders
        for sp in SPLITS:
            video_labels = []
            landmarks_from_videos = []
            for gloss in org_gloss_rank.keys():
                # Iteration over each video
                for video_path in os.listdir(os.path.join(config.VIDEOS_PATH, sp, gloss)):
                    video_landmarks = []
                    cap = cv2.VideoCapture(os.path.join(config.VIDEOS_PATH, sp, gloss, video_path))
                    # Loop until the end of the video
                    counter = 0
                    ret = True
                    while ret:
                        ret, frame = cap.read()
                        if ret:
                            # There are 4 landmarks that we are interested in: face-mesh, pose, rigth-hand and left-hand 
                            # so we are going to extract them with the holistic process method
                            results = holistic.process(frame)
                            if results is None:
                                print("Something is wrong w/ mediapipe... Exiting.")
                                cap.release()
                                return
                            
                            # Once we get the landmarks, we insert them in an array
                            video_landmarks.append(extract_landmarks_to_np(results))
                            counter += 1

                    cap.release()
                    # with all the frames landmarks extracted (videos), we append them 
                    # to a list of video landmarks
                    landmarks_from_videos.append(video_landmarks)
                    video_labels.append(gloss)
                print("INFO: ", gloss, "label processed for", sp, "split")
            
            X[sp] = landmarks_from_videos
            Y[sp] = video_labels
            print("INFO: ", sp, "split processed.")
    
    if save:
        # with all correctly stored, we save the file in a pickl file.
        print("Data stored in dictionary, saving in pickel file under data/npy_videos folder...")
        save_dict(X, X_PICK_FILE_PATH)
        save_dict(Y, Y_PICK_FILE_PATH)

    print("Removing the auxiliar folders...")
    for sp in SPLITS:
        shutil.rmtree(os.path.join(config.VIDEOS_PATH, sp), onerror=rm_error_info)
        print(os.path.join(config.VIDEOS_PATH, sp), "removed.")

    X_tens = from_dict_to_tensor(X)
    Y_enc = encode_labels(Y)

    return X_tens, Y_enc


if __name__ == '__main__':
    
    load = False
    if load:
        X, Y = load_transform_save_dataset(top_k=20, save=True)
    else:
        X = load_dict(X_PICK_FILE_PATH)
        Y = load_dict(Y_PICK_FILE_PATH)
    
    X_train = X['train']
    X_test = X['test']
    X_val = X['val']

    Y_train = Y['train']
    Y_test = Y['test']
    Y_val = Y['val']

    print("X_train: ", X_train[0])
    print("Y_train: ", Y_train[0])