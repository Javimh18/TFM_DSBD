from models import MediaPipeLSTM
import torch
from mp_loaders import load_dataset_from_pickle

if __name__ == '__main__':

    dataset = "top_10"
    X, y, label_map = load_dataset_from_pickle(dataset=dataset)

    X_train, X_val, X_test = X['train'], X['val'], X['test']
    y_train, y_val, y_test = X['train'], X['val'], X['test']

    n_frames = X_train[0].shape[0]
    n_landmarks = X_train[0].shape[1]

    print(f"N_frames by video: {n_frames}")
    print(f"N_landmmarks by frame: {n_landmarks}")
