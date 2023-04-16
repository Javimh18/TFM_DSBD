import os

# Define de inputs of the data and the metadata. In this case, the metadata is 
# represented via JSON file in the data directory
DATA_PATH = f"./data"
METADATA_PATH = os.path.join(DATA_PATH, 'WLASL_v0.3.json')

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
LM_PER_VIDEO = 23

EXTENSION = '.mp4'
SPLITS = ['train', 'val', 'test']
PCKL_PATH = os.path.join(DATA_PATH, "pckl_files")
X_PICK_FILE_NAME = 'npy_db_x.pkl'
Y_PICK_FILE_NAME = 'npy_db_y.pkl'
LABELS_MAP_PICK_FILE_NAME = 'labels_map.pkl'