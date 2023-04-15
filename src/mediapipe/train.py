from model import MediaPipeLSTM
from torch.utils.data import TensorDataset, DataLoader
import torch
from mp_loaders import load_dataset_from_pickle
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from tqdm.auto import tqdm
import torch.nn.functional as F
from train_test_functions import train_step, test_step, eval_model, accuracy_fn

from config import BATCH_SIZE, NUM_EPOCHS, NUM_WORKERS, INIT_LR
from mp_funs import FACEMESH_LANDMARKS, POSE_LANDMARKS, HAND_LANDMARKS

TOTAL_LANDMARKS = FACEMESH_LANDMARKS + POSE_LANDMARKS + 2*HAND_LANDMARKS

if __name__ == '__main__':

    dataset = "top_10"
    X_dict, y_dict, label_map = load_dataset_from_pickle(dataset=dataset)

    # we unpack the tensors as a dictionary, where each entry is a split in the dataset
    X_train, X_val, X_test = X_dict['train'], X_dict['val'], X_dict['test']
    y_train, y_val, y_test = y_dict['train'], y_dict['val'], y_dict['test']

    # casting tensors from double dtype to float
    X_train = X_train.type(torch.FloatTensor)
    X_val = X_val.type(torch.FloatTensor)
    X_test = X_test.type(torch.FloatTensor)

    # We wrappimg into a PyTorch Dataset using TensorDataset
    train_data = TensorDataset(X_train, y_train)
    val_data = TensorDataset(X_train, y_train)
    test_data = TensorDataset(X_train, y_train)

    # We load the Dataset as Dataloader for better performance in training stage
    train_dataloader = DataLoader(dataset=train_data, 
                                  batch_size=BATCH_SIZE, 
                                  shuffle=True, 
                                  pin_memory=True,
                                  num_workers=NUM_WORKERS,
                                  drop_last=True)
    
    val_dataloader = DataLoader(dataset=val_data, 
                                batch_size=BATCH_SIZE, 
                                shuffle=True, 
                                pin_memory=True,
                                num_workers=NUM_WORKERS,
                                drop_last=True)
    
    test_dataloader = DataLoader(dataset=test_data, 
                                 batch_size=BATCH_SIZE, 
                                 shuffle=True, 
                                 pin_memory=True,
                                 num_workers=NUM_WORKERS,
                                 drop_last=True)
    
    # setting the device accordingly to the hardware
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # extract the number of frames per video
    n_frames = list(next(iter(train_dataloader))[0].shape)[1]
    # instantiate the model
    model = MediaPipeLSTM(input_dim=TOTAL_LANDMARKS, output_dim=10, hidden_dim=100)#.to(device)

    """
    X, y = next(iter(train_dataloader))
    print(torch.flatten(X, start_dim=1).unsqueeze(1).shape)
    print(model(torch.flatten(X, start_dim=1).unsqueeze(1)))
    """

    # define loss function and optimizer
    loss_fn = CrossEntropyLoss()
    optimizer = Adam(params=model.parameters(), lr=INIT_LR)

    # training loop 
    for epoch in range(NUM_EPOCHS):
        train_step(model, train_dataloader, loss_fn, optimizer, accuracy_fn, device)

    
        
    
        

    
        
    
        
    

    
    





