import keras
import numpy as np

class DataGenerator(keras.utils.Sequence):
    def __init__(self, df, indexes, L_win, NCATS, batch_size):
        self.data = df
        self.batch_size = batch_size
        self.ind = indexes
        self.L_win = L_win
        self.NCATS = NCATS
        
    def __len__(self):
        return int(np.floor(len(self.ind) / self.batch_size))
    
    def __getitem__(self, idx):
        batch_ind = self.ind[idx * self.batch_size:(idx + 1) * self.batch_size]
        Ck = batch_ind[:, 0]
        month = batch_ind[:, 1] - 1
        ind_x = batch_ind[:, -(self.L_win + 1) : -1]
        ind_y = batch_ind[:, -1]

        X = self.data[ind_x, : ]
        Y = self.data[ind_y, :]
        Y = np.where(self.data[ind_y,:], 1, 0)
        X = X.reshape(self.batch_size, self.L_win, self.NCATS)
        Y = Y.reshape(self.batch_size, self.NCATS) 
        return [X, Ck, month], Y