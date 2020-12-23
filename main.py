import pandas as pd
import matplotlib.pyplot as plt

import EvaluateResults
from ClassesIdentification import ClassesIdentification

class SpendingPrediction:
    __lstm_params = {'window length': 4,
                     'number of filters': 64,
                     'learning rate': 0.001,
                     'batch size': 64,
                     'number of epochs': 10,
                     '1st month of test period': 7,
                     'target category': 63}  #MCC 63 corresponds to restaurants category
    __bilstm_params = {'number of timesteps': 6,
                       'number of filters': 20,
                       'number of epochs': 30,
                       'batch size': 32}
    
    def set_lstm_model(self, L_win = 4, NFILTERS = 64, lr = 0.001, BATCH_SIZE = 64, 
                       NB_EPOCH = 10, month_test = 7, pred_cat = 63):
        self.__lstm_params['window length'] = L_win
        self.__lstm_params['number of filters'] = NFILTERS
        self.__lstm_params['learning rate'] = lr
        self.__lstm_params['batch size'] = BATCH_SIZE
        self.__lstm_params['number of epochs'] = NB_EPOCH
        self.__lstm_params['1st month of test period'] = month_test
        self.__lstm_params['target category'] = pred_cat
        
    def set_bilstm_model(self, n_timesteps = 6, BiNFILTERS = 20, BiNB_EPOCH = 30,
                         BiBATCH_SIZE = 32):
        self.__bilstm_params['number of timesteps'] = n_timesteps
        self.__bilstm_params['number of filters'] = BiNFILTERS
        self.__bilstm_params['number of epochs'] = BiNB_EPOCH
        self.__bilstm_params['batch size'] = BiBATCH_SIZE
        
    def infer(self, df):
        model = ClassesIdentification(df, self.__lstm_params['window length'], self.__lstm_params['number of filters'],
                               self.__lstm_params['learning rate'], self.__lstm_params['batch size'],
                               self.__lstm_params['number of epochs'], self.__lstm_params['1st month of test period'],
                               self.__lstm_params['target category'], self.__bilstm_params['number of timesteps'],
                               self.__bilstm_params['number of filters'], self.__bilstm_params['number of epochs'],
                               self.__bilstm_params['batch size'])
        y_pred, y_true = model.y_pred, model.y_true
        A, B = model.A, model.B
        y_pred_A, y_true_A, y_pred_B, y_true_B = model.answers_for_classes(A, B)
        pred_A, pred_B = model.BiLSTM_model()
        y_pred_est_A, y_true_est_A, y_pred_est_B, y_true_est_B = model.answers_for_classes(pred_A, pred_B)
        
        return y_pred, y_true, y_pred_A, y_true_A, y_pred_B, y_true_B, y_pred_est_A, y_true_est_A, y_pred_est_B, y_true_est_B
    
plt.rcParams["figure.figsize"] = [7, 5]
colors = ['k','g','r','g', 'r']
linestyles = ['-', '-', '-', '-', '-']

data = pd.read_csv("data/train_set.csv",index_col=None)
SP = SpendingPrediction()
SP.set_lstm_model(L_win = 4, NFILTERS = 64, lr = 0.001, BATCH_SIZE = 64, NB_EPOCH = 10, 
                  month_test = 7, pred_cat = 63)
SP.set_bilstm_model(n_timesteps = 6, BiNFILTERS = 20, BiNB_EPOCH = 30, BiBATCH_SIZE = 32)
y_pred_LSTM, y_true_LSTM, y_pred_A, y_true_A, y_pred_B, y_true_B, y_pred_est_A, y_true_est_A, y_pred_est_B, y_true_est_B = SP.infer(df = data)

err_RNN = EvaluateResults.make_err_df(y_true_LSTM, y_pred_LSTM)
err_RNN.name = 'All data'

err_RNN_A = EvaluateResults.make_err_df(y_true_A, y_pred_A)
err_RNN_A.name = 'High predictability class'

err_RNN_B = EvaluateResults.make_err_df(y_true_B, y_pred_B)
err_RNN_B.name = 'Low predictability class'

err_RNN_est_A = EvaluateResults.make_err_df(y_true_est_A, y_pred_est_A)
err_RNN_est_A.name = 'High predictability class (estimated)'

err_RNN_est_B = EvaluateResults.make_err_df(y_true_est_B, y_pred_est_B)
err_RNN_est_B.name = 'Low predictability class (estimated)'

larger_elements = [element for element in y_true_LSTM if element > 0]
freq = len(larger_elements) / len(y_true_LSTM)
plt.plot([0, 1], [freq, freq], label = 'Event frequency', linewidth=2, linestyle='-', color = 'b')
recalls = EvaluateResults.plot_PRC([err_RNN, err_RNN_A, err_RNN_B], 0.5, colors, linestyles)
plt.show()
print(recalls)

plt.plot([0, 1], [freq, freq], label = 'Event frequency', linewidth=2, linestyle='-', color = 'b')
recalls = EvaluateResults.plot_PRC([err_RNN, err_RNN_est_A, err_RNN_est_B], 0.5, colors, linestyles)
plt.show()
print(recalls)


