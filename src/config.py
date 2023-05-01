import os

# Define de inputs of the data and the metadata. In this case, the metadata is 
# represented via JSON file in the data directory
DATA_PATH = f"./data"
METADATA_PATH = os.path.join(DATA_PATH, 'labels.csv')
VIDEOS_PATH = os.path.join(DATA_PATH, 'videos')

# define the path to the base output directory
BASE_OUTPUT = os.path.join(DATA_PATH, "output")
# Define the path to the output serialized model
MODEL_PATH = os.path.join(BASE_OUTPUT, "sign_detector.h5")

# Initialization parameters such as initial learning rate, number of epochs to train 
# or the initial batch size
INIT_LR = 1e-3
NUM_EPOCHS = 200
BATCH_SIZE = 48
NUM_WORKERS = os.cpu_count()
LM_PER_VIDEO = 10

EXTENSION = '.mp4'
SPLITS = ['train', 'val', 'test']
PCKL_PATH = os.path.join("pckl_files")
X_PICK_FILE_NAME = 'npy_db_x.pkl'
Y_PICK_FILE_NAME = 'npy_db_y.pkl'
LABELS_MAP_PICK_FILE_NAME = 'labels_map.pkl'
EXTRACTED_FRAMES_PICK_FILENAME = 'extracted_frames.pkl'

FACEMESH_LANDMARKS = 468*3 # 468 points with 3 coordinates (x, y, z) each 
POSE_LANDMARKS = 33*4 # 33 points with 4 coordinates (x, y, z and visibility) each
HAND_LANDMARKS = 21*3 # 21 points with 3 coordinates (x, y, z) each

# TOTAL_LANDMARKS = FACEMESH_LANDMARKS + POSE_LANDMARKS + 2*HAND_LANDMARKS
TOTAL_LANDMARKS = POSE_LANDMARKS + 2*HAND_LANDMARKS + FACEMESH_LANDMARKS