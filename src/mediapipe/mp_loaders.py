import mediapipe as mp
from sklearn import preprocessing
import torch
import numpy as np
import os
import cv2
from mp_funs import extract_landmarks_to_np, FACEMESH_LANDMARKS, POSE_LANDMARKS, HAND_LANDMARKS
from utils import save_dict, load_dict
from math import floor
from config import SPLITS, X_PICK_FILE_NAME, Y_PICK_FILE_NAME, LABELS_MAP_PICK_FILE_NAME, DATA_PATH, PCKL_PATH

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities that will be useful for action representation


def encode_labels(labels):
    new_labels = {}
    le = preprocessing.LabelEncoder()
    le.fit(labels['train'])

    for sp in SPLITS:
        new_labels[sp] = torch.tensor(le.transform(labels[sp]))

    le_mapping = dict(zip(le.classes_, le.transform(le.classes_)))

    return new_labels, le_mapping 


def save_dataset(X_tens, Y_enc, le_mapping, dataset:str):

    if not os.path.exists(os.path.join(PCKL_PATH, dataset)):
        os.makedirs(os.path.join(PCKL_PATH, dataset), exist_ok=True)

    save_dict(X_tens, os.path.join(PCKL_PATH, dataset, X_PICK_FILE_NAME))
    save_dict(Y_enc, os.path.join(PCKL_PATH, dataset, Y_PICK_FILE_NAME))
    save_dict(le_mapping, os.path.join(PCKL_PATH, dataset, LABELS_MAP_PICK_FILE_NAME))


def load_dataset_from_pickle(dataset:str):
    X_tens = load_dict(os.path.join(PCKL_PATH, dataset, X_PICK_FILE_NAME))
    Y_enc = load_dict(os.path.join(PCKL_PATH, dataset, Y_PICK_FILE_NAME))
    le_mapping = load_dict(os.path.join(PCKL_PATH, dataset, LABELS_MAP_PICK_FILE_NAME))
    return X_tens, Y_enc, le_mapping


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
    dims = {}
    for sp in SPLITS:
        split = X[sp]
        for video in split:
            diff = max_len - len(video)
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
                    for i in range(0, int(diff/2)):
                        video.append(np.zeros(n_frame_lm))
                        video.insert(i, np.zeros(n_frame_lm))
        
    for sp in SPLITS:
        # Retrieve the dimensions from the tensors
        dims[sp] = (len(X[sp]), max_len, n_frame_lm)
        # flatten the nested list of np.arrays 
        X[sp] = np.concatenate(X[sp]).ravel()
        
    # Now that the number of frames between videos are matched, we cast them into tensors
    for sp in SPLITS:
        X[sp] = torch.tensor(X[sp]).reshape(dims[sp])

    return X


def transform_to_mediapipe_tensors_dataset(top_k=200, 
                                           save=False):

    print(f"Applying mediapipe transformations to the top_{top_k} dataset...")
    dataset = f"top_{top_k}"
    X = {}
    Y = {}
    # declaration of the holistic model in media pipe
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        # we iterate through the folders
        for sp in SPLITS:
            video_labels = []
            landmarks_from_videos = []
            for gloss in os.listdir(os.path.join(DATA_PATH, dataset, sp)):
                # Iteration over each video
                for video_name in os.listdir(os.path.join(DATA_PATH, dataset, sp, gloss)):
                    video_landmarks = []
                    cap = cv2.VideoCapture(os.path.join(DATA_PATH, dataset, sp, gloss, video_name))
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
    
    print("Transforming the data to Pytorch Tensors...")
    X_tens = from_dict_to_tensor(X)
    Y_enc, le_mapping = encode_labels(Y)

    if save:
        # with all correctly stored, we save the file in a pickl file.
        print("Data stored in dictionary, saving in pickel file under data/pckl_files folder...")
        save_dataset(X_tens, Y_enc, le_mapping, dataset)
    else:
        return X_tens, Y_enc, le_mapping
    
if __name__ == '__main__':
    transform_to_mediapipe_tensors_dataset(top_k=10, save=True)