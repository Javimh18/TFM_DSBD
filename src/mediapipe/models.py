import torch
from torch import nn
from math import floor
from torch.autograd import Variable 
from mp_loaders import load_dataset_from_pickle

class MediaPipeLSTM(nn.Module):
    def __init__(self, n_frames, n_landmarks, n_classes, hidden_dim):
        super(MediaPipeLSTM, self).__init__()
        # Dimension of the hidden layer (short-term and output)
        self.hidden_dim = n_classes
        # number of landmarks extracted from the videos
        self.n_landmarks = n_landmarks
        # Number of frames per video
        self.n_frames = n_frames
        # Hidden dim of the STM channel
        self.hidden_dim = hidden_dim
        # Initialize hidden state
        self.hidden_st = (torch.randn(1, 1, hidden_dim))

        # define the LSTM, which takes all the landmarks, from each frame of the video
        self.lstm = nn.LSTM(input_size = n_landmarks, hidden_size = hidden_dim, 
                            num_layers = 1, batch_first = True)

        # now we map, the hidden_dim to the n_classes our model has to predict
        self.hidden_to_dense = nn.Linear(hidden_dim, n_classes + floor(n_classes/3))
        self.dense_to_fc = nn.Linear(hidden_dim, n_classes)
        self.fc_to_relu = nn.ReLU()

        self.layers_stack = nn.Sequential([

        ])


    def forward(self, x):

        lstm_out, (h_n, c_n) = self.lstm(x, self.hidden_st)
        relu1 = self.fc_to_relu(fc)
        dense = self.hidden_to_dense(relu1)
        relu2 = self.fc_to_relu(dense)
        fc = self.dense_to_fc(relu1)
        logits = self.fc_to_relu(fc)

        return logits


    





