import numpy as np
import statistics

from LSTM_predictor import LSTM_predictor

class ClassesInferring(LSTM_predictor):
    def __init__(self, df, L_win, NFILTERS, lr, BATCH_SIZE, NB_EPOCH, month_test,
                 pred_cat):
        LSTM_predictor.__init__(self, df, L_win, NFILTERS, lr, BATCH_SIZE, NB_EPOCH,
                                month_test, pred_cat)
        self.CE = self.count_coefficient()
        self.A, self.B = self.predictability_classes()

    def count_coefficient(self):
        ind_test_loc, ind_train_loc = self.ind_test.copy(), self.ind_train.copy()
        ind_train_loc['target'] = self.table.values[ind_train_loc.values[:, -1], self.table.columns.get_loc(self.pred_cat)]
        ind_train_loc['target'] = np.where(ind_train_loc['target'], 1, 0)
        ind_test_loc['target'] = self.table.values[ind_test_loc.values[:, -1], self.table.columns.get_loc(self.pred_cat)]
        ind_test_loc['target'] = np.where(ind_test_loc['target'], 1, 0)
        ind_test_loc['predicted_prob'] = np.zeros(len(ind_test_loc))
        ind_test_loc.iloc[:len(self.y_pred), ind_test_loc.columns.get_loc('predicted_prob')] = self.y_pred
        ind_test_loc = ind_test_loc[:len(self.y_pred)]
        ind_test_loc['num'] = abs(ind_test_loc['target'] - ind_test_loc['predicted_prob'])
        num = ind_test_loc.groupby('id')['num'].sum()
        den_n = ind_test_loc.groupby('id')['target'].count()
        CE = num / den_n
        CE = 1 - CE
        return CE     
    
    def predictability_classes(self):
        A, B = [], []
        for c_id, ce in self.CE.items():
            if (ce > statistics.median(self.CE)):
                A.append(c_id)
            else:
                B.append(c_id)
        return A, B   