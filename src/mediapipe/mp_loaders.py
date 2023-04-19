import mediapipe as mp
from sklearn import preprocessing
import torch
import numpy as np
from numpy.random import default_rng
import os
import cv2
from tqdm import tqdm 
from mp_funs import extract_landmarks_to_np, FACEMESH_LANDMARKS, POSE_LANDMARKS, HAND_LANDMARKS
from utils import save_dict, load_dict
from config import SPLITS, X_PICK_FILE_NAME, Y_PICK_FILE_NAME, LABELS_MAP_PICK_FILE_NAME, VIDEOS_PATH, PCKL_PATH, LM_PER_VIDEO, EXTRACTED_FRAMES_PICK_FILENAME

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities that will be useful for action representation

rng = default_rng()

def save_dataset(X_tens, Y_enc, le_mapping, extracted_frames, framework: str):

    if not os.path.exists(os.path.join(PCKL_PATH, framework)):
        os.makedirs(os.path.join(PCKL_PATH, framework), exist_ok=True)

    save_dict(X_tens, os.path.join(PCKL_PATH, framework, X_PICK_FILE_NAME))
    save_dict(Y_enc, os.path.join(PCKL_PATH, framework, Y_PICK_FILE_NAME))
    save_dict(le_mapping, os.path.join(PCKL_PATH, framework, LABELS_MAP_PICK_FILE_NAME))
    save_dict(le_mapping, os.path.join(PCKL_PATH, extracted_frames, EXTRACTED_FRAMES_PICK_FILENAME))


def load_dataset_from_pickle(framework: str):
    X_tens = load_dict(os.path.join(PCKL_PATH, framework, X_PICK_FILE_NAME))
    Y_enc = load_dict(os.path.join(PCKL_PATH, framework, Y_PICK_FILE_NAME))
    le_mapping = load_dict(os.path.join(PCKL_PATH, framework, LABELS_MAP_PICK_FILE_NAME))
    extracted_frames = load_dict(os.path.join(PCKL_PATH, framework, EXTRACTED_FRAMES_PICK_FILENAME))
    return X_tens, Y_enc, le_mapping, extracted_frames

def label_dict(labels):
    new_labels = {}
    le = preprocessing.LabelEncoder()
    le.fit(labels['train'])

    le_mapping = dict(zip(le.classes_, le.transform(le.classes_)))

    for sp in SPLITS:
        new_labels[sp] = le.fit_transform(labels[sp])

    return new_labels, le_mapping


def from_dict_to_pytorch_tensor(X, labels):
    max_len = LM_PER_VIDEO
    n_frame_lm = FACEMESH_LANDMARKS + POSE_LANDMARKS + 2*HAND_LANDMARKS

    new_labels = {}
    le = preprocessing.LabelEncoder()
    le.fit(labels['train'])

    le_mapping = dict(zip(le.classes_, le.transform(le.classes_)))

    for sp in SPLITS:
        new_labels[sp] = torch.tensor(le.transform(labels[sp]))

    dims = {}     
    for sp in SPLITS:
        # Retrieve the dimensions from the tensors
        dims[sp] = (len(X[sp]), max_len, n_frame_lm)
        # flatten the nested list of np.arrays 
        X[sp] = np.concatenate(X[sp]).ravel()
        
    # Now that the number of frames between videos are matched, we cast them into tensors
    for sp in SPLITS:
        X[sp] = torch.tensor(X[sp]).reshape(dims[sp])

    return X, new_labels, le_mapping


def transform_to_mediapipe_tensors_dataset(save=False,
                                           framework="tf",
                                           path_to_subset=None):
    
    if path_to_subset is None:
        path = VIDEOS_PATH
    else:
        path = path_to_subset
    
    X = {}
    Y = {}
    extracted_frames = {}
    # declaration of the holistic model in media pipe
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        # we iterate through the folders
        for sp in SPLITS:
            video_labels = []
            landmarks_from_videos = []
            extracted_frames_from_videos = []
            for gloss in tqdm(os.listdir(os.path.join(path, sp))):
                # Iteration over each video
                for video_name in os.listdir(os.path.join(path, sp, gloss)):
                    video_landmarks = []
                    cap = cv2.VideoCapture(os.path.join(path, sp, gloss, video_name))
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
                    # with all the frames landmarks extracted (videos), we randomly extract 10 frames,
                    # and append them to a list that contains of the landmarks for each frame of each video
                    extracted_n_lm_idx = sorted(rng.choice(len(video_landmarks), size=LM_PER_VIDEO, replace=False))
                    extracted_n_lm = []
                    
                    for i in range(len(video_landmarks)):
                        if i in extracted_n_lm_idx:
                            extracted_n_lm.append(video_landmarks[i])

                    landmarks_from_videos.append(extracted_n_lm)
                    video_labels.append(gloss)
                    extracted_frames_from_videos.append(extracted_n_lm)
            if framework == 'pytorch':
                X[sp] = landmarks_from_videos
                Y[sp] = video_labels
                extracted_frames[sp] = np.array(extracted_frames_from_videos)
            elif framework == 'tf':
                X[sp] = np.array(landmarks_from_videos)
                Y[sp] = np.array(video_labels)
                extracted_frames[sp] = np.array(extracted_frames_from_videos)
            print("INFO: ", sp, "split processed.")
    
    if framework == "pytorch":
        print("Transforming the data to Pytorch Tensors...")
        X_tens, Y_enc, le_mapping = from_dict_to_pytorch_tensor(X, Y)
    elif framework == "tf":
        print("Transforming the data to TF Tensors...")
        X_tens = X
        Y_enc, le_mapping = label_dict(Y)

    if save:
        # with all correctly stored, we save the file in a pickl file.
        print(f"Data stored in dictionary, saving in pickel file under data/pckl_files/{framework} folder...")
        save_dataset(X_tens, Y_enc, le_mapping, extracted_frames, framework)
    else:
        return X_tens, Y_enc, le_mapping, extracted_frames, 
    
if __name__ == '__main__':
    transform_to_mediapipe_tensors_dataset(save=True, framework="tf", path_to_subset='data/subset_10_lsa_64')