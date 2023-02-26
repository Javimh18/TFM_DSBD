import os

# Define de inputs of the data and the metadata. In this case, the metadata is 
# represented via JSON file in the data directory
BASE_PATH = 'data'
VIDEOS_PATH = os.path.join(BASE_PATH, 'videos')
METADATA_PATH = os.path.join(BASE_PATH, 'WLASL_v0.3.json')

# define the path to the base output directory
BASE_OUTPUT = "output"
# Define the path to the output serialized model
MODEL_PATH = os.path.join(BASE_OUTPUT, "sign_detector.h5")
#PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot.png"])

# Initialization parameters such as initial learning rate, number of epochs to train 
# or the initial batch size

INIT_LR = 1e-4
NUM_EPOCHS = 30
BATCH_SIZE = 32