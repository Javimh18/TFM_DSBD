import torch
from tqdm.auto import tqdm
import torch.nn.functional as F
from config import BATCH_SIZE


# Calculate accuracy - TP/(TP + TN) - out of a 100 example, what percentage does our model get right
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct/len(y_true)) * 100
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
        
        # we flatten each video into an array of landmarks
        X = torch.flatten(X, start_dim=1).unsqueeze(1)

        # send the data to the device
        X, y = X.to(device), y.to(device)

        # get predictions
        pred_logits = model(X)
        y_pred = torch.argmax(F.log_softmax(pred_logits, dim=1), dim=1)

        # compute the loss
        loss = loss_fn(pred_logits, y)
        train_loss += loss.item()

        # compute the accuracy
        train_acc += accuracy_fn(y_true=y,
                                 y_pred=y_pred)

        # reset the gradient of the optimizer
        optimizer.zero_grad()

        # Performing backpropagation
        loss.backward()

        # Updating the parameters
        optimizer.step()

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
            test_logits = model(X)
            test_pred = torch.argmax(F.log_softmax(test_logits, dim=1), dim=1)
            # 2. Compute the loss
            test_loss += loss_fn(test_logits, y)   
            # 3. Compute the accuracy
            test_acc += accuracy_fn(y_true=y, 
                                    y_pred=test_pred) 

        # Divide total test loss by length of train dataloader
        test_loss /= len(data_loader)
        # Compute the test acc average per batch
        test_acc /= len(data_loader)

        print(f"Test Loss: {test_loss:.5f} | Test acc {test_acc:.2f} %")


def eval_model(model: torch.nn.Module,
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
            # 1. Forward pass
            val_logits = model(X)
            test_pred = torch.argmax(F.log_softmax(val_logits, dim=1), dim=1)

            # Accumulate the loss and accuracy values per batch
            loss += loss_fn(val_logits, y)
            acc += accuracy_fn(y_true=y,
                               y_pred=test_pred)
            
        # Scale loss and acc to find avg loss/acc
        loss /= len(data_loader)
        acc /= len(data_loader)

    return {"model_name": model.__class__.__name__,
            "model_loss": loss.item(),
            "model_acc": acc}