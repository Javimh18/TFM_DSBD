import mediapipe as mp
from sklearn import preprocessing
import numpy as np
from numpy.random import default_rng
import os
import cv2
from tqdm import tqdm 
from mp_funs import extract_landmarks_to_np
from utils import save_dict, load_dict
from config import SPLITS, X_PICK_FILE_NAME, Y_PICK_FILE_NAME, LABELS_MAP_PICK_FILE_NAME, VIDEOS_PATH, PCKL_PATH, LM_PER_VIDEO, EXTRACTED_FRAMES_PICK_FILENAME

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities that will be useful for action representation

rng = default_rng()

def save_dataset(X_tens: dict, Y_enc: dict, le_mapping: dict, path_to_save:str = None):

    if path_to_save is None:
        print("Path to save empty. Please specify a path to where save the landmarks.")
        return

    if not os.path.exists(os.path.join(path_to_save)):
        os.makedirs(os.path.join(path_to_save), exist_ok=True)

    save_dict(X_tens, os.path.join(path_to_save, X_PICK_FILE_NAME))
    save_dict(Y_enc, os.path.join(path_to_save, Y_PICK_FILE_NAME))
    save_dict(le_mapping, os.path.join(path_to_save, LABELS_MAP_PICK_FILE_NAME))


def label_dict(labels):
    new_labels = {}
    le = preprocessing.LabelEncoder()
    le.fit(labels['train'])

    le_mapping = dict(zip(le.classes_, le.transform(le.classes_)))

    for sp in SPLITS:
        new_labels[sp] = le.fit_transform(labels[sp])

    return new_labels, le_mapping


def transform_to_mediapipe_tensors_dataset(save = False,
                                           path_to_save = None,
                                           path_to_subset = None):
    
    if path_to_subset is None:
        path = VIDEOS_PATH
    else:
        path = path_to_subset
    
    X = {}
    Y = {}
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
            
            X[sp] = np.array(landmarks_from_videos)
            Y[sp] = np.array(video_labels)
            print("INFO: ", sp, "split processed.")
    
        print("Transforming the data to TF Tensors...")
        X_tens = X
        Y_enc, le_mapping = label_dict(Y)

    if save:
        # with all correctly stored, we save the file in a pickl file.
        print(f"Data stored in dictionary, saving in pickel file under {path_to_save} folder...")
        save_dataset(X_tens, Y_enc, le_mapping, path_to_save)
    else:
        return X_tens, Y_enc, le_mapping
    
if __name__ == '__main__':
    transform_to_mediapipe_tensors_dataset(save=True, path_to_save="./data/subset_10_lsa_64/pickl_files", path_to_subset='data/subset_10_lsa_64')