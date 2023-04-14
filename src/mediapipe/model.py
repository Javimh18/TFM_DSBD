import torch
from torch import nn
from math import floor
import torch.nn.functional as F

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
        self.hidden_st = (torch.randn(1, hidden_dim), torch.randn(1, hidden_dim))

        # define the LSTM, which takes all the landmarks, from each frame of the video
        self.lstm = nn.LSTM(input_size = n_landmarks, hidden_size = hidden_dim, 
                            num_layers = 1, batch_first = True)

        # now we map, the hidden_dim to the n_classes our model has to predict
        self.hidden_to_dense = nn.Linear(hidden_dim, n_classes + floor(n_classes/3))
        self.dense_to_fc = nn.Linear(n_classes + floor(n_classes/3), n_classes)
        self.fc_to_relu = nn.ReLU()

    def forward(self, x):

        device = "cuda" if torch.cuda.is_available() else "cpu"

        lstm_out, self.hidden_st = self.lstm(x, (self.hidden_st[0].to(device), self.hidden_st[1].to(device)))
        relu1 = self.fc_to_relu(lstm_out)
        dense = self.hidden_to_dense(relu1)
        relu2 = self.fc_to_relu(dense)
        fc = self.dense_to_fc(relu2)
        logits = self.fc_to_relu(fc)
        pred_prob = F.log_softmax(logits, dim=1)

        return pred_prob


    





