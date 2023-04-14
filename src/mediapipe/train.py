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
    X, y, label_map = load_dataset_from_pickle(dataset=dataset)

    # we unpack the tensors as a dictionary, where each entry is a split in the dataset
    X_train, X_val, X_test = X['train'], X['val'], X['test']
    y_train, y_val, y_test = y['train'], y['val'], y['test']

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
                                  num_workers=NUM_WORKERS)
    
    val_dataloader = DataLoader(dataset=val_data, 
                                batch_size=BATCH_SIZE, 
                                shuffle=True, 
                                pin_memory=True,
                                num_workers=NUM_WORKERS)
    
    test_dataloader = DataLoader(dataset=test_data, 
                                 batch_size=BATCH_SIZE, 
                                 shuffle=True, 
                                 pin_memory=True,
                                 num_workers=NUM_WORKERS)
    
    print(f"Length of train_dataloader: {len(train_dataloader)} batches of {BATCH_SIZE}...")
    print(f"Length of test_dataloader: {len(test_dataloader)} batches of {BATCH_SIZE}...")
    
    # setting the device accordingly to the hardware
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # extract the number of frames per video
    X, y = next(iter(train_dataloader))
    n_frames = list(X.shape)[0]
    # instantiate the model
    model = MediaPipeLSTM(n_frames=n_frames, n_landmarks=TOTAL_LANDMARKS, n_classes=10, hidden_dim=100).to(device)

    y = model(X)
    print(f"Y: {y}")
    print(f"Shape of Y: {y.shape}")


    # define loss function and optimizer
    loss_fn = CrossEntropyLoss()
    optimizer = Adam(params=model.parameters(), lr=INIT_LR)

    # training loop 
    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch} of training step")
        train_step(model=model,
                   data_loader=train_dataloader,
                   loss_fn=loss_fn,
                   optimizer=optimizer,
                   accuracy_fn=accuracy_fn,
                   device=device)
    """    
        test_step(model=model,
                   data_loader=val_data,
                   loss_fn=loss_fn,
                   accuracy_fn=accuracy_fn,
                   device=device)
        

    model_results = eval_model(model=model,
                    data_loader=val_data,
                    loss_fn=loss_fn,
                    accuracy_fn=accuracy_fn,
                    device=device)
    
    print(model_results)
    """
        
    
        

    
        
    
        
    

    
    





