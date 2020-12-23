import pandas as pd
import csv

class DataProcessing:
    def __init__(self, df, L_win, month_test):
        self.df = df
        self.preproc_data = self.preprocessing()
        self.table = self.features_aggregation()
        self.L_win = L_win
        self.NCATS = self.count_categories()
        self.ind_test, self.ind_train = DataProcessing.train_test_split(self, month_test)

    def preprocessing(self):
        #df is a set of transactions with the following fields:
        #amount, customer_id, mcc, transaction_date
        with open('data/mcc2big.csv') as csvfile:
            reader = csv.reader(csvfile)
            next(reader, None)
            catdict = {int(rows[0]): int(rows[1]) for rows in reader}
        self.df['MCC87'] = self.df['mcc'].map(catdict)
        self.df.dropna(inplace = True, subset = ['MCC87'])
        self.df['MCC87'] = self.df['MCC87'].astype('int')
        self.df['transaction_date'] = pd.to_datetime(self.df['transaction_date'], infer_datetime_format = True)
        self.df['MONTH'] = self.df['transaction_date'].apply(lambda date: date.month)
        return self.df

    def features_aggregation(self):
        data_sum = self.preproc_data.groupby(['customer_id', 'MCC87', 'MONTH'], as_index = False)['amount'].sum()
        data_sum['COUNT'] = self.preproc_data.groupby(['customer_id', 'MCC87', 'MONTH']).size().reset_index().iloc[:, -1]
        labels, uniques = pd.factorize(data_sum['customer_id'])
        data_sum['id'] = labels
        table_N = data_sum.pivot_table(index = ['id','MONTH'], columns = 'MCC87', values = 'COUNT', fill_value = 0).reset_index()
        return table_N
    
    def window(self, in_group, ind_ar):
        istart = 0
        istop = self.L_win + 1   
        group = in_group.sort_values()    
        indices = group.index
        gr = group
        while istop <= len(group):
            m_start = gr.iloc[istart]
            m_stop = gr.iloc[istop - 1]
            if (m_stop - m_start) == self.L_win:
                add_data = [group.name, group.iloc[istop - 1]]           
                indxs = add_data + [it for it in indices[istart:istop]]
                ind_ar.append(indxs)
            istart += 1
            istop += 1
        return ind_ar   
    
    def train_test_split(self, month_test):
        ind_ar = []
        self.table.groupby('id')['MONTH'].apply(lambda x: self.window(x, ind_ar))
        df_indxs = pd.DataFrame(ind_ar, columns=['id','last_month'] + list(range(self.L_win + 1)))
        ind_test = df_indxs[df_indxs['last_month'] >= month_test]
        ind_train = df_indxs[df_indxs['last_month'] < month_test]
        return ind_test, ind_train
    
    def count_categories(self):
        NCATS = self.table.shape[1] - 2
        return NCATS