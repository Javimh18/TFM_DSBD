import numpy as np
import pandas as pd

class Leaderboard:
    def __init__(self, path_to_lb: str='./data/leaderboard.csv'):
        self.path_to_lb = path_to_lb
        self.lb = pd.read_csv(path_to_lb)

    def update_leaderboard(self, history, test_results, model_name, model_type, subset_size):
        # check if the type of the model exists. In that case update that entry on
        # the leader board. If not, create new entry to the dataframe
        if (True in self.lb['model_type'].isin(model_type)) and (True in self.lb['model_name'].isin(model_name)) and (True in self.lb['subset_size'].isin(subset_size)):
            row = self.lb.loc[(self.lb['model_type'] == model_type) and (self.lb['model_name'] == model_name)]
            val_acc_epoch = np.argmax(history.history['val_accuracy']) 
            val_acc_max = history.history['val_accuracy'][val_acc_epoch]
            if row['val_acc'] < val_acc_max:
                row['train_acc'] = history.history['accuracy'][val_acc_epoch]
                row['val_acc'] = val_acc_max
                row['test_acc'] = test_results[1]
                row['train_loss'] = history.history['loss'][val_acc_epoch]
                row['val_loss'] = history.history['val_loss'][val_acc_epoch]
                row['test_loss'] = test_results[0]
                row['epoch'] = val_acc_epoch + 1
                row['path_to_model'] = f'src/{model_type}/bestmodels/best_{model_name}_{val_acc_epoch+1}_{val_acc_max}_model.h5'
                self.lb.loc[self.lb['type'] == model_type] = row

        else:
            val_acc_epoch = np.argmax(history.history['val_accuracy'])
            train_acc_max = history.history['accuracy'][val_acc_epoch]
            val_acc_max = history.history['val_accuracy'][val_acc_epoch]
            test_acc_max = test_results[1]
            train_loss_max = history.history['loss'][val_acc_epoch]
            val_loss_max =  history.history['val_loss'][val_acc_epoch]
            test_loss_max = test_results[0]
            epoch = val_acc_epoch + 1
            path_to_model =  f'src/{model_type}/bestmodels/best_{model_name}_{epoch}_{val_acc_max}_model.h5'
            subset_size = subset_size

            new_row = {
                'model_type': model_type,
                'model_name': model_name,
                'train_acc': train_acc_max,
                'val_acc': val_acc_max,
                'test_acc': test_acc_max,
                'train_loss': train_loss_max,
                'val_loss': val_loss_max,
                'test_loss': test_loss_max,
                'epoch': epoch,
                'path_to_model': path_to_model,
                'subset_size': subset_size
            }

            # we add the new row to the dataframe and sort the dataframe based on the val_acc
            self.lb.append(new_row, ignore_index=True)\
                   .sort_values('val_acc', axis=0, ascending=False)
            
        # we save the updated version of the leaderboard
        self.lb.to_csv(self.path_to_lb, header=True, index=False, mode='w+')


    def create_leaderboard(path_to_save='data/leaderboard'):
        df = pd.DataFrame(columns=['model_type', 'model_name', 'train_acc', 'val_acc', 'test_acc', 'train_loss', 'val_loss', 'test_loss', 'epoch', "path_to_model", "subset_size"])
        df.to_csv(path_to_save)

    
            
