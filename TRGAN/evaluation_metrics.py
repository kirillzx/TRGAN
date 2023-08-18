import pandas as pd
import numpy as np
import copy
import random
from scipy.special import rel_entr, kl_div
from scipy.spatial.distance import jensenshannon
from scipy.stats import kstest, ks_2samp, wasserstein_distance
import scipy.stats as sts
from sdmetrics.single_column import TVComplement
from sdv.metadata import SingleTableMetadata
from sdmetrics.single_table import NewRowSynthesis
from sdmetrics.column_pairs import ContingencySimilarity
from sklearn.metrics import f1_score, recall_score, roc_auc_score
from IPython.display import display

import keras
from keras.layers import Input, Dense, Activation, Dropout,Lambda, LSTM, BatchNormalization
from keras.models import Model
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras.optimizers.legacy import Adam
from keras import backend as K

'''
NUMERICAL
'''

def evaluate_numerical(data_array, index):
    mean_values = list(map(lambda x: x.mean(), data_array))
    std_values = list(map(lambda x: x.std(), data_array))
    kurt_values = list(map(lambda x: x.kurtosis(), data_array))
    skew_values = list(map(lambda x: x.skew(), data_array))
    js_values = list(map(lambda x: jensenshannon(np.sort(data_array[0]), np.sort(x)), data_array))
    ks_values = list(map(lambda x: ks_2samp(data_array[0], x)[0], data_array))
    wd_values = list(map(lambda x: wasserstein_distance(data_array[0], x), data_array))

    metrics = np.array([mean_values, std_values, kurt_values, skew_values, js_values, ks_values, wd_values]).T
    res_df = pd.DataFrame(metrics, columns=['Mean', 'Std', 'Kurtosis', 'Skewness', 'D_JS', 'KS2test', 'Wassertein distance'],\
                           index=index)

    return res_df

def evaluate_numerical_cashflow(data_array, index):
    mean_values = list(map(lambda x: x.mean(), data_array))
    std_values = list(map(lambda x: x.std(), data_array))
    kurt_values = list(map(lambda x: x.kurtosis(), data_array))
    skew_values = list(map(lambda x: x.skew(), data_array))
    js_values = list(map(lambda x: jensenshannon(np.sort(abs(data_array[0])), np.sort(abs(x))), data_array))
    ks_values = list(map(lambda x: ks_2samp(data_array[0], x)[0], data_array))
    wd_values = list(map(lambda x: wasserstein_distance(data_array[0], x), data_array))

    metrics = np.array([mean_values, std_values, kurt_values, skew_values, js_values, ks_values, wd_values]).T
    res_df = pd.DataFrame(metrics, columns=['Mean', 'Std', 'Kurtosis', 'Skewness', 'D_JS', 'KS2test', 'Wassertein distance'],\
                           index=index)

    return res_df

'''
Categorical
'''

def evaluate_categorical(data_array, index, data_cont_array):
    tv_values = list(map(lambda x: TVComplement.compute(real_data=data_array[0], synthetic_data=x), data_array))
    # new_rows_values =  list(map(lambda x: NewRowSynthesis.compute(data_array[0], synthetic_data=x,\
    #                             metadata=metadata, numerical_match_tolerance=0.4, synthetic_sample_size=10_000), data_array))
    
    cont_sim_value = list(map(lambda x: ContingencySimilarity.compute(real_data=data_cont_array[0], synthetic_data=x), data_cont_array))
    val_counts = list(map(lambda x: int(x.value_counts().shape[0]), data_array))

    js_divergence = list(map(lambda x: jensenshannon(data_array[0], x), data_array))

    res_df = pd.DataFrame(np.array([tv_values, cont_sim_value, val_counts, js_divergence]).T,\
                        columns=['Total Variation', 'Contingency Similarity', 'Values count', 'D_JS'], index=index)

    return res_df


def evaluate_new_rows(data_array, index, metadata):

    new_rows = list(map(lambda x: NewRowSynthesis.compute(
                    real_data=data_array[0],
                    synthetic_data=x,
                    metadata=metadata,
                    numerical_match_tolerance=0.0005,
                    synthetic_sample_size=10_000), data_array))
    
    res_df = pd.DataFrame(np.array([new_rows]).T,\
                        columns=['New Rows Synthesis'], index=index)

    return res_df


def utility_metrics_ml(data: pd.DataFrame):
    data['transaction_date'] = pd.to_datetime(data['transaction_date'], infer_datetime_format=True)
    data['MONTH'] = data['transaction_date'].apply(lambda date: date.month)
    data['YEAR'] = data['transaction_date'].apply(lambda date: date.year)
    
    data_sum = data.groupby(['customer', 'mcc','MONTH'], as_index=False)['amount'].sum()
    data_sum['COUNT'] = data.groupby(['customer', 'mcc','MONTH']).size().reset_index().iloc[:,-1]
    labels, uniques = pd.factorize(data_sum['customer'])
    data_sum['id'] = labels
    
    table_N = data_sum.pivot_table(index=['id', 'MONTH'], columns='mcc', values='COUNT',fill_value=0).reset_index()
    table_V = data_sum.pivot_table(index=['id', 'MONTH'], columns='mcc', values='amount',fill_value=0).reset_index()
    
    global L_win, ar
    L_win = 4
    ar = []
    
    table_N.groupby('id')['MONTH'].apply(window)
    df_indxs= pd.DataFrame(ar, columns=['id', 'last_month']+list(range(L_win+1)))

    month_test = 10
    ind_test = df_indxs[df_indxs['last_month'] == month_test]
    ind_train = df_indxs[df_indxs['last_month'] < month_test]
    NCATS = table_N.shape[1] - 2
    
    OPTIM = Adam(lr=0.001)
    NFILTERS = 128
    
    model_RNN = create_model(NCATS, NFILTERS, OPTIM)
    BATCH_SIZE = 64
    NB_EPOCH = 20
    g_train = DataGenerator(table_N.values[:,2:], ind_train.values, BATCH_SIZE, NCATS)
    g_test = DataGenerator(table_N.values[:,2:], ind_test.values, BATCH_SIZE, NCATS)
    model_RNN.fit_generator(generator=g_train, validation_data=g_test,epochs=NB_EPOCH, verbose=0)
    
    y_pred = model_RNN.predict_generator(generator=g_test)
    y_true = np.vstack([g_test[i][1] for i in range(len(g_test))])
    
    TEST_CAT = 3
    def make_err_df(y_true, y_pred):
        return pd.DataFrame(np.vstack((y_true,y_pred)).transpose(), columns=['y_true', 'y_pred'])

    err_RNN = make_err_df(y_true[:,TEST_CAT],y_pred[:,TEST_CAT])
    err_RNN.name = 'RNN'
    
    df = pd.DataFrame(np.array([[f1_score(err_RNN['y_true'], np.where(err_RNN['y_pred'] < 0.5, 0, 1))],\
                        [recall_score(err_RNN['y_true'], np.where(err_RNN['y_pred'] < 0.5, 0, 1))],\
                        [roc_auc_score(err_RNN['y_true'], err_RNN['y_pred'])]]).T, columns=['F_1', 'Recall', 'AUC'])
    
    return df


def evaluate_utility(data: list, index_names: list):
    res_df = pd.DataFrame(columns=['F_1', 'Recall', 'AUC'])
    
    for i in data:
        df = utility_metrics_ml(i)
        res_df = pd.concat([res_df, df])
        
    res_df.index = index_names
    
    display(res_df)
    
    
    
    
def window(in_group):
    istart = 0
    istop = L_win+1   
    group = in_group.sort_values()    
    indices = group.index
    gr = group
    while istop <= len(group):
        m_start = gr.iloc[istart]
        m_stop = gr.iloc[istop - 1]
        if (m_stop - m_start) == L_win:
            add_data = [group.name,group.iloc[istop - 1]]           
            indxs = add_data+[it for it in indices[istart:istop]]
            ar.append(indxs)
        istart += 1
        istop += 1
    
    
def my_mean_squared_error(y_true, y_pred):
    return K.mean(K.clip(y_true, 0, 1)*K.square(y_pred - y_true), axis=-1)

class DataGenerator(keras.utils.Sequence):
    def __init__(self, df, indexes, batch_size , NCATS):
        self.data = df
        self.batch_size = batch_size
        self.ind = indexes
        self.ncts = NCATS
        
    def __len__(self):
        return int(np.floor(len(self.ind) / self.batch_size))
    
    def __getitem__(self, idx):
        batch_ind = self.ind[idx * self.batch_size:(idx + 1) * self.batch_size]
        Ck = batch_ind[:,0]
        month = batch_ind[:, 1]-1
        ind_x = batch_ind[:, -(L_win+1):-1]
        ind_y = batch_ind[:,-1]

        X = self.data[ind_x,:]
        Y = self.data[ind_y,:]
        Y = np.where(self.data[ind_y,:], 1, 0)
        X = X.reshape(self.batch_size, L_win, self.ncts)
        Y = Y.reshape(self.batch_size, self.ncts) 
        return [X, Ck, month], Y

def create_model(NCATS, NFILTERS, OPTIM):
    inp = Input(shape=(L_win,NCATS))
    inp_ck = Input(shape=(1,))
    inp_m = Input(shape=(1,))
    
    lay = LSTM(NFILTERS)(inp)
    trg_clf = Dense(NCATS, activation='sigmoid')(lay)

    model_clf = Model(inputs=[inp,inp_ck,inp_m], outputs=trg_clf)
    model_clf.compile(loss='binary_crossentropy',optimizer=OPTIM,metrics=['accuracy'])
    
    return model_clf