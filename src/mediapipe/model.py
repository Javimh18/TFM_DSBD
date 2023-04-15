import torch
from torch import nn
from math import floor
from config import BATCH_SIZE

# Idea taken from: https://towardsdatascience.com/from-a-lstm-cell-to-a-multilayer-lstm-network-with-pytorch-2899eb5696f3
class MediaPipeLSTM(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(MediaPipeLSTM, self).__init__()

        self.in_dim = input_dim
        self.out_dim = output_dim
        self.hidden_dim = hidden_dim
        self.seq_len = 155

        self.lstmcell_1 = nn.LSTMCell(input_dim, hidden_dim)
        self.lstmcell_2 = nn.LSTMCell(hidden_dim, hidden_dim)
        self.stack_fc_nn = nn.Sequential(
            nn.Linear(in_features=hidden_dim, out_features=output_dim+30),
            nn.ReLU(),
            nn.Linear(in_features=output_dim+30, out_features=output_dim),
            nn.ReLU()
        )


    def forward(self, x):

        # batch_size x hidden_size
        hidden_state = torch.zeros(x.size(0), self.hidden_dim)
        cell_state = torch.zeros(x.size(0), self.hidden_dim)
        hidden_state_2 = torch.zeros(x.size(0), self.hidden_dim)
        cell_state_2 = torch.zeros(x.size(0), self.hidden_dim)
        hidden_state_3 = torch.zeros(x.size(0), self.hidden_dim)
        cell_state_3 = torch.zeros(x.size(0), self.hidden_dim)
        
        # weights initialization
        torch.nn.init.xavier_normal_(hidden_state)
        torch.nn.init.xavier_normal_(cell_state)
        torch.nn.init.xavier_normal_(hidden_state_2)
        torch.nn.init.xavier_normal_(cell_state_2)
        torch.nn.init.xavier_normal_(hidden_state_3)
        torch.nn.init.xavier_normal_(cell_state_3)

        # device = "cuda" if torch.cuda.is_available() else "cpu"
        # we view the whole video as sequences of 155 frames of 1662 landmarks (155, BATCH_SIZE, 1662)
        x = x.view(self.seq_len, x.size(0), -1)

        for i in range(self.seq_len):
            hidden_state, cell_state = self.lstmcell_1(x[i], (hidden_state, cell_state))
            hidden_state_2, cell_state_2 = self.lstmcell_2(hidden_state, (hidden_state_2, cell_state_2))
            hidden_state_3, cell_state_3 = self.lstmcell_2(hidden_state, (hidden_state_3, cell_state_3))

        out = self.stack_fc_nn(hidden_state_3)

        return out

    





