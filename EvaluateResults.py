import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
        
def make_err_df(y_true, y_pred):
    return pd.DataFrame(np.vstack((y_true, y_pred)).transpose(), columns=['y_true','y_pred'])
    
def plot_PRC(errs, lim, colors, linestyles):
    recalls = {}
    for im, err in enumerate(errs):
        ytrue = err['y_true'].values 
        ypred = err['y_pred'].values

        precision, recall, thr = precision_recall_curve(ytrue, ypred)
        area = auc(recall, precision)
        clr = colors[im]
        lns = linestyles[im]
        plt.plot(recall, precision, linewidth = 2, color = clr, label = err.name, linestyle = lns)
                
        ind = np.argmin(recall > lim)    
        if ind < len(thr):
            r = recall[ind]
        recalls[err.name] = r
        
        plt.xlabel('Recall', fontsize = 20)
        plt.ylabel('Precision', fontsize = 20)
        plt.ylim(0, 1)
        plt.title('PR curve',size = 20,weight = 'bold')
        plt.legend()
    return recalls, precision[ind], thr[ind], area