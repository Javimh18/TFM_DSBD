from model import MediaPipeLSTM
from torch.utils.data import TensorDataset, DataLoader
import torch
from mp_loaders import load_dataset_from_pickle
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from tqdm.auto import tqdm

from config import BATCH_SIZE, NUM_EPOCHS, NUM_WORKERS, INIT_LR
from mp_funs import FACEMESH_LANDMARKS, POSE_LANDMARKS, HAND_LANDMARKS

TOTAL_LANDMARKS = FACEMESH_LANDMARKS + POSE_LANDMARKS + 2*HAND_LANDMARKS

# Calculate accuracy - TP/(TP + TN) - out of a 100 example, what percentage does our model get right
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct/len(y_pred)) * 100
    return acc

def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device: torch.device):
    
    train_loss, train_acc = 0, 0
    # set the model to train mode
    model.train()
    for batch, (X, y) in enumerate(data_loader):

        # send the data to the device
        X, y = X.to(device), y.to(device)

        # get predictions
        pred_probs = model(X)
        y_pred = torch.argmax(pred_probs, dim=1)

        # compute the loss
        loss = loss_fn(pred_probs, y)
        train_loss += loss

        # compute the accuracy
        train_acc += accuracy_fn(y_true=y,
                                 y_pred=y_pred)

        # reset the gradient of the optimizer
        optimizer.zero_grad()

        # Performing backpropagation
        loss.backward()

        # Updating the parameters
        optimizer.step()

        if batch % 10 == 0:
            print(f"Looked at {batch * len(X)} / {len(train_dataloader.dataset)} samples.")

        # Divide total train loss by length of train dataloader
        train_loss /= len(data_loader)
        train_acc /= len(data_loader)

        print(f"Train Loss: {train_loss:.5f} | Train acc {train_acc:.2f} %")

    return


def test_step(model: torch.nn.Module,
              data_loader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              accuracy_fn,
              device: torch.device
              ):
    ## testing 
    test_loss, test_acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X_test, y_test in data_loader:
            X, y = X_test.to(device), y_test.to(device)
            # 1. Forward pass
            test_pred = model(X)
            # 2. Compute the loss
            test_loss += loss_fn(test_pred, y)   
            # 3. Compute the accuracy
            test_acc += accuracy_fn(y_true=y, 
                                    y_pred=test_pred.argmax(dim=1)) 

        # Divide total test loss by length of train dataloader
        test_loss /= len(data_loader)
        # Compute the test acc average per batch
        test_acc /= len(data_loader)

        print(f"Test Loss: {test_loss:.5f} | Test acc {test_acc:.2f} %")


def eval_model_device(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               accuracy_fn,
               device):
    """Returns a dictionary containing the results of model predicting the data_loader."""
    loss, acc = 0,0
    model.eval()
    with torch.inference_mode():
        for X, y in tqdm(data_loader):
            # Make predictions
            X, y = X.to(device), y.to(device)
            y_pred = model(X)

            # Accumulate the loss and accuracy values per batch
            loss += loss_fn(y_pred, y)
            acc += accuracy_fn(y_true=y,
                               y_pred=y_pred.argmax(dim=1))
            
        # Scale loss and acc to find avg loss/acc
        loss /= len(data_loader)
        acc /= len(data_loader)

    return {"model_name": model.__class__.__name__,
            "model_loss": loss.item(),
            "model_acc": acc}

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

    # extract the number of frames per video
    X, y = next(iter(train_data))
    n_frames = list(X.shape)[0]

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
    
    # setting the device accordingly to the hardware
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # instantiate the model
    model = MediaPipeLSTM(n_frames=n_frames, n_landmarks=TOTAL_LANDMARKS, n_classes=10, hidden_dim=100).to(device)

    # define loss function and optimizer
    loss_fn = CrossEntropyLoss()
    optimizer = Adam(params=model.parameters(), lr=INIT_LR)

    # training loop 
    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch} of training step")
        train_step(model=model,
                   data_loader=train_data,
                   loss_fn=loss_fn,
                   optimizer=optimizer,
                   accuracy_fn=accuracy_fn,
                   device=device)
        
        test_step(model=model,
                   data_loader=val_data,
                   loss_fn=loss_fn,
                   accuracy_fn=accuracy_fn,
                   device=device)
        

    model_results = eval_model_device(model=model,
                    data_loader=val_data,
                    loss_fn=loss_fn,
                    accuracy_fn=accuracy_fn,
                    device=device)
    
    print(model_results)
        
    
        

    
        
    
        
    

    
    





