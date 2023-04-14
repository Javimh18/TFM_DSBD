import torch
from torch import nn

class MediaPipeLSTM(nn.Module):
    def __init__(self, n_frames, n_landmarks, n_classes, hidden_dim):
        super().__init__()
        # Dimension of the hidden layer (short-term and output)
        self.hidden_dim = n_classes
        # number of landmarks extracted from the videos
        self.n_landmarks = n_landmarks
        # Number of frames per video
        self.n_frames = n_frames
        # Hidden dim of the STM channel
        self.hidden_dim = hidden_dim
        # initialize the hidden state as a random tensor of n_landmarks by n_frames
        self.hidden_st = torch.randn(1, n_frames, n_landmarks)

        # define the LSTM, which takes all the landmarks, from each frame of the video
        self.lstm = nn.LSTM(n_landmarks * n_frames, hidden_dim)

        # now we map, the hidden_dim to the n_classes our model has to predict
        self.hidden_to_class = nn.Linear(hidden_dim, n_classes)


    def forward(self, video_landmarks):

        lstm_out, self.hidden_st = self.lstm(video_landmarks, self.hidden_st)
        logits = self.hidden_to_class(lstm_out)
        probs = torch.softmax(logits, dim=1)

        return probs
    




