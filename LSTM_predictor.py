from keras.layers import Input, LSTM, Dense
from keras.models import Model
from keras.optimizers import Adam

import numpy as np

from DataGenerator import DataGenerator
from processing import DataProcessing

class LSTM_predictor(DataProcessing):
    def __init__(self, df, L_win, NFILTERS, lr, BATCH_SIZE, NB_EPOCH, month_test,
                 pred_cat):
        DataProcessing.__init__(self, df, L_win, month_test)
        # self.L_win = L_win
        # self.table = DataProcessing.features_aggregation(self)
        # self.NCATS = DataProcessing.count_categories(self)
        # self.ind_test, self.ind_train = DataProcessing.train_test_split(self, month_test)
        self.NFILTERS = NFILTERS
        self.OPTIM = Adam(lr)
        self.BATCH_SIZE = BATCH_SIZE
        self.NB_EPOCH = NB_EPOCH
        self.pred_cat = pred_cat
        self.y_pred, self.y_true = self.train_and_predict()
    
    def create_model(self):
        inp = Input(shape=(self.L_win, self.NCATS))
        inp_ck = Input(shape = (1, ))
        inp_m = Input(shape = (1, ))
    
        lay = LSTM(self.NFILTERS, return_sequences = True)(inp)
        lay2 = LSTM(self.NFILTERS)(lay)
        trg_clf = Dense(self.NCATS, activation = 'sigmoid')(lay2)

        model_clf = Model(inputs = [inp, inp_ck, inp_m], outputs = trg_clf)
        model_clf.compile(loss = 'binary_crossentropy', optimizer = self.OPTIM, metrics = ['accuracy']) 
    
        return model_clf
    
    def train_and_predict(self):
        g_train = DataGenerator(self.table.values[:, 2:], self.ind_train.values, self.L_win, self.NCATS, self.BATCH_SIZE)
        g_test = DataGenerator(self.table.values[:, 2:], self.ind_test.values, self.L_win, self.NCATS, self.BATCH_SIZE)
        model_RNN = self.create_model()
        model_RNN.fit_generator(generator = g_train, validation_data = g_test, 
                                          epochs = self.NB_EPOCH, verbose = 1)
        model_RNN.save_weights('LSTM.h5')
        y_pred = model_RNN.predict_generator(generator = g_test)
        y_true = np.vstack([g_test[i][1] for i in range(len(g_test))])
        ind_cat = self.table.columns.get_loc(self.pred_cat) - 2
        return y_pred[:, ind_cat], y_true[:, ind_cat]     
    
    def answers_for_classes(self, classA, classB):
        ind_test_A = self.ind_test[self.ind_test['id'].isin(classA)]
        ind_test_B = self.ind_test[self.ind_test['id'].isin(classB)]
        y_pred_A, y_true_A = self.get_answers_for_classes(ind_test_A)
        y_pred_B, y_true_B = self.get_answers_for_classes(ind_test_B)
        ind_cat = self.table.columns.get_loc(self.pred_cat) - 2
        return y_pred_A[:, ind_cat], y_true_A[:, ind_cat], y_pred_B[:, ind_cat], y_true_B[:, ind_cat]
    
    def get_answers_for_classes(self, ind_test):
        model_RNN = self.create_model()
        model_RNN.load_weights("LSTM.h5")
        g_test = DataGenerator(self.table.values[:,2:], ind_test.values, self.L_win, self.NCATS, self.BATCH_SIZE)
        y_pred = model_RNN.predict_generator(generator=g_test)
        y_true = np.vstack([g_test[i][1] for i in range(len(g_test))])
        return y_pred, y_true