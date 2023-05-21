import numpy as np
import pandas as pd
import os

class Leaderboard:
    def __init__(self, path_to_lb: str='./data/leaderboard.csv'):
        self.path_to_lb = path_to_lb
        if os.path.exists(path_to_lb):
            self.lb = pd.read_csv(path_to_lb, header=0)
        else:
            self.lb = pd.DataFrame(columns=['model_type', 'model_name', 'train_acc', 'val_acc', 'test_acc', 'train_loss', 'val_loss', 'test_loss', 'epoch', "path_to_model", "subset_size"])

    def update_leaderboard(self, history, test_results, model_name, model_type, subset_size):
        # check if the type of the model exists. In that case update that entry on
        # the leader board. If not, create new entry to the dataframe
        if self.lb['model_type'].eq(model_type).any() and self.lb['model_name'].eq(model_name).any() and self.lb['subset_size'].eq(subset_size).any():
            row = self.lb.loc[self.lb['model_type'].eq(model_type) & self.lb['model_name'].eq(model_name) & self.lb['subset_size'].eq(subset_size)]
            val_acc_epoch = np.argmax(history.history['val_accuracy']) 
            val_acc_max = history.history['val_accuracy'][val_acc_epoch]
            print(row['val_acc'].item())
            if row['val_acc'].item() < val_acc_max:
                updated_row = {
                    'model_type': model_type,
                    'model_name': model_name,
                    'train_acc': float("{:.2f}".format(history.history['accuracy'][val_acc_epoch])),
                    'val_acc': float("{:.2f}".format(val_acc_max)),
                    'test_acc': float("{:.2f}".format(test_results[1])),
                    'train_loss': float("{:.2f}".format(history.history['loss'][val_acc_epoch])),
                    'val_loss': float("{:.2f}".format(history.history['val_loss'][val_acc_epoch])),
                    'test_loss': float("{:.2f}".format(test_results[0])),
                    'epoch': val_acc_epoch + 1,
                    'path_to_model': f'src/{model_type}/bestmodels/best_{model_name}_{val_acc_epoch+1}_{val_acc_max:.2f}_model.h5',
                    'subset_size': subset_size
                }
                
                self.lb.drop(self.lb[self.lb['model_type'].eq(model_type) & self.lb['model_name'].eq(model_name)].index, axis=0, inplace=True)
                self.lb.loc[len(self.lb)] = updated_row
                self.lb.sort_values('val_acc', axis=0, ascending=False)


        else:
            val_acc_epoch = np.argmax(history.history['val_accuracy'])
            train_acc_max = history.history['accuracy'][val_acc_epoch]
            val_acc_max = history.history['val_accuracy'][val_acc_epoch]
            test_acc_max = test_results[1]
            train_loss_max = history.history['loss'][val_acc_epoch]
            val_loss_max =  history.history['val_loss'][val_acc_epoch]
            test_loss_max = test_results[0]
            epoch = val_acc_epoch + 1
            path_to_model =  f'src/{model_type}/bestmodels/best_{model_name}_{epoch}_{val_acc_max:.2f}_model.h5'
            subset_size = subset_size

            new_row = {
                'model_type': model_type,
                'model_name': model_name,
                'train_acc': float("{:.2f}".format(train_acc_max)),
                'val_acc': float("{:.2f}".format(val_acc_max)),
                'test_acc': float("{:.2f}".format(test_acc_max)),
                'train_loss': float("{:.2f}".format(train_loss_max)),
                'val_loss': float("{:.2f}".format(val_loss_max)),
                'test_loss': float("{:.2f}".format(test_loss_max)),
                'epoch': epoch,
                'path_to_model': path_to_model,
                'subset_size': subset_size
            }

            # we add the new row to the dataframe and sort the dataframe based on the val_acc
            self.lb.loc[len(self.lb)] = new_row
            self.lb.sort_values('val_acc', axis=0, ascending=False)
            
        # we save the updated version of the leaderboard
        self.lb.to_csv(self.path_to_lb, header=True, index=False, mode='w+')