import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Bidirectional, LSTM, Dense

from ClassesInferring import ClassesInferring

class ClassesIdentification(ClassesInferring):
    def __init__(self, df, L_win, NFILTERS, lr, BATCH_SIZE, NB_EPOCH, month_test, 
                 pred_cat, n_timesteps, BiNFILTERS, BiNB_EPOCHS, BiBATCH_SIZE):
        ClassesInferring.__init__(self, df, L_win, NFILTERS, lr, BATCH_SIZE, NB_EPOCH,
                                  month_test, pred_cat)
        self.n_timesteps = n_timesteps
        self.BiNFILTERS = BiNFILTERS
        self.BiNB_EPOCHS = BiNB_EPOCHS
        self.BiBATCH_SIZE = BiBATCH_SIZE
        
    def train_test_split_for_class_identification(self):
        ind = np.arange(1, 7)
        ind_test = np.arange(7, 13)
        X_train = self.table[self.table['MONTH'].isin(ind)]
        X_train = X_train[X_train['id'].isin(self.CE.keys())]
        X_test = self.table[self.table['MONTH'].isin(ind_test)]
        X_test = X_test[X_test['id'].isin(self.CE.keys())]
        num_months = X_test.groupby('id').size()
        valid_months = [c_id for c_id, num in num_months.items() if num > 5]
        X_test = X_test[X_test['id'].isin(valid_months)]
        num_months_train = X_train.groupby('id').size()
        valid_months_train = [c_id for c_id, num in num_months_train.items() if num > 5]
        X_train = X_train[X_train['id'].isin(valid_months_train)]
        test_ids = np.unique(X_test['id'])
        return X_train, X_test, test_ids

    def reshape_train_and_test(self):
        list_X, list_X_test = [], []
        X_train, X_test, test_ids = self.train_test_split_for_class_identification()
        un_ids = np.unique(X_train['id'])
        Y = np.zeros(len(un_ids))
        un_ids_test = np.unique(X_test['id'])
        for c_id in un_ids:
            cur = X_train[X_train['id'] == c_id][self.pred_cat].values
            list_X.append(cur)
        for i in range(len(un_ids)):
            if un_ids[i] in self.A:
                Y[i] = 0
            elif un_ids[i] in self.B:
                Y[i] = 1
        for i in range(len(list_X)):
            list_X[i] = list_X[i][-6:]
        X = np.vstack(list_X)
        X = X.reshape(len(list_X), self.n_timesteps, 1)
        Y = pd.get_dummies(Y).values
        Y = Y.reshape(len(list_X), 2)
        for c_id in un_ids_test:
            cur = X_test[X_test['id'] == c_id][self.pred_cat].values
            list_X_test.append(cur)
        X_test_n = np.vstack(list_X_test)
        X_test_n = X_test_n.reshape(len(list_X_test), self.n_timesteps, 1)
        return X, Y, X_test_n, test_ids
        
    def BiLSTM_model(self):
        X, Y, X_test_n, test_ids = self.reshape_train_and_test()
        model = Sequential()
        model.add(Bidirectional(LSTM(self.BiNFILTERS, input_shape=(self.n_timesteps, 1), 
                                     return_sequences = False)))
        model.add(Dense(2, activation = 'softmax'))
        model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
        model.fit(X, Y, epochs = self.BiNB_EPOCHS, batch_size = self.BiBATCH_SIZE, 
                  verbose=2)
        yhat = model.predict_classes(X_test_n)
        pred_A, pred_B = [], []
        for i in range(len(yhat)):
            if (yhat[i] == 0):
                pred_A.append(test_ids[i])
            else:
                pred_B.append(test_ids[i])
        return pred_A, pred_B