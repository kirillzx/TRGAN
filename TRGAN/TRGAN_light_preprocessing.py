import pandas as pd
import numpy as np
import copy
import random
import sys

import torch
from torch import optim
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from scipy.optimize import minimize
from scipy.stats import wasserstein_distance, entropy
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from rdt.transformers.numerical import ClusterBasedNormalizer, GaussianNormalizer
from rdt.transformers.categorical import UniformEncoder
from typing import Union, TypeVar
import scipy.stats as sts



'''
CATEGORICAL FEATURES
'''

#forward transformation

def create_categorical_embeddings(data_init: pd.DataFrame, feat_names: list) -> Union[np.array, None, list]:
    freq_enc = []
    new_features = []
    data = copy.deepcopy(data_init)

    for i in range(len(feat_names)):
        enc = UniformEncoder()
        customer_enc = enc.fit_transform(data, column=feat_names[i])[feat_names[i]].values
        
        freq_enc.append(enc)
        new_features.append(customer_enc)

    embeddings = np.array(new_features).T
    embeddings = embeddings.astype(float)

    scaler = MinMaxScaler((-1, 1))
    embeddings = scaler.fit_transform(embeddings)

    return embeddings, scaler, freq_enc


# inverse transformation

def inverse_categorical_embeddings(cat_embeddings: np.array, feat_names: list, scaler, freq_enc: list) -> np.array:
    emb_recovered = scaler.inverse_transform(cat_embeddings)
  
    decoded_array = []
    for i in range(len(feat_names)):
        temp = freq_enc[i].reverse_transform(pd.DataFrame(emb_recovered[:, i], columns=[feat_names[i]]))
        decoded_array.append(temp.values)

    decoded_array = np.array(decoded_array)
    emb_recovered = decoded_array.T[0]
    
    return emb_recovered




'''
NUMERICAL FEATURES
'''

#forward transformation

def create_numerical_embeddings(data_init, cont_features, type_scale='GaussianNormalize', max_clusters=15, weight_threshold=0.005):
    data = copy.deepcopy(data_init)

    if type_scale == 'CBNormalize':
        amt = data[cont_features]

        data_normalized = []
        data_component = []
        scaler = []

        for i in range(len(cont_features)):
            cbn =  ClusterBasedNormalizer(learn_rounding_scheme=True, enforce_min_max_values=True,\
                                        max_clusters=max_clusters, weight_threshold=weight_threshold)
            data1 = cbn.fit_transform(amt, column=cont_features[i])
            data_normalized.append(data1[cont_features[i]+'.normalized'])
            data_component.append(data1[cont_features[i]+'.component'])
            scaler.append(cbn)

        data[cont_features] = np.vstack(data_normalized).T
        components = np.vstack(data_component).T
        scaler.append(components)
        X_cont = data[cont_features].values

    elif type_scale == 'Standardize':
        scaler = []
        scaler_std = StandardScaler()
        data[cont_features] = scaler_std.fit_transform(data[cont_features].values)
        X_cont = data[cont_features].values
        scaler.append(scaler_std)
        
    elif type_scale == 'GaussianNormalize':
        embeddings = copy.deepcopy(data[cont_features])
        gaus_tr = []
        
        for col in cont_features:
            transformer = GaussianNormalizer(learn_rounding_scheme=True, enforce_min_max_values=True)
            transformer.reset_randomization()
            embeddings[col] = transformer.fit_transform(data, column=[col])[col]
            gaus_tr.append(transformer)
            
        embeddings = np.array(embeddings)
        
        scaler = MinMaxScaler((-1, 1))
        embeddings = scaler.fit_transform(embeddings)    
    

    else:
        print('Choose preprocessing scheme for continuous features. Available: GaussianNormalize, CBNormalize and Standardize')

    # return data[cont_features].values, scaler
    gaus_tr = np.array(gaus_tr, dtype=object)
    
    return embeddings, scaler, gaus_tr


#inverse transformation

def inverse_numerical_embeddings(num_embeddings: np.array, feat_names: list, scaler, num_transf: list) -> np.array:
    emb_recovered = scaler.inverse_transform(num_embeddings)
    # synth_data_scaled_cl = synth_data_scaled_cl.astype(int)
    # synth_data_scaled_cl = np.where(synth_data_scaled_cl < 0, 0, synth_data_scaled_cl)

    decoded_array = []
    for i in range(len(feat_names)):
        num_transf[i].reset_randomization()
        temp = num_transf[i].reverse_transform(pd.DataFrame(emb_recovered[:, i], columns=[feat_names[i]]))
        decoded_array.append(temp.values)

    decoded_array = np.array(decoded_array)
    emb_recovered = decoded_array.T[0]
    
    return emb_recovered


def learn_rounding_digits(data):
    MAX_DECIMALS = sys.float_info.dig - 1
    roundable_data = data[~(np.isinf(data.astype(float)) | pd.isna(data))]

    # Doesn't contain numbers
    if len(roundable_data) == 0:
        return None

    # Doesn't contain decimal digits
    if ((roundable_data % 1) == 0).all():
        return 0

    # Try to round to fewer digits
    if (roundable_data == roundable_data.round(MAX_DECIMALS)).all():
        for decimal in range(MAX_DECIMALS + 1):
            if (roundable_data == roundable_data.round(decimal)).all():
                return decimal

    return None


def learn_round(data, num_feat_names, time_feature):
    if time_feature != '':
        data[num_feat_names[:-3]] = data[num_feat_names[:-3]].apply(lambda x: round(x, sys.float_info.dig - 1))

        round_array = []
        for col in num_feat_names[:-3]:
            round_array.append(learn_rounding_digits(data[col]))
            
    else:
        data[num_feat_names] = data[num_feat_names].apply(lambda x: round(x, sys.float_info.dig - 1))

        round_array = []
        for col in num_feat_names:
            round_array.append(learn_rounding_digits(data[col]))

    return round_array


def make_round(data, round_array, num_feat_names, time_feature):
    data = copy.deepcopy(data)
    
    if time_feature != '':
        for i in range(len(num_feat_names)-3):
            if round_array[i] == 0:
                data[num_feat_names[i]] = data[num_feat_names[i]].apply(lambda x: round(x))
            else:
                data[num_feat_names[i]] = data[num_feat_names[i]].apply(lambda x: round(x, round_array[i]))
            
    else:
        for i in range(len(num_feat_names)):
            if round_array[i] == 0:
                data[num_feat_names[i]] = data[num_feat_names[i]].apply(lambda x: round(x))
            else:
                data[num_feat_names[i]] = data[num_feat_names[i]].apply(lambda x: round(x, round_array[i]))
            
    return data



def prob_int_transform(data: np.array) -> np.array:
    scaler_01 = MinMaxScaler()
    temp = scaler_01.fit_transform(data) 
    temp = np.where(temp > 0, temp, temp + 1e-5)
    temp = np.where(temp < 1, temp, temp - 1e-5)
    
    tr_arr = []
    for i in range(data.shape[1]):
        tr_arr.append(sts.norm.ppf(temp[:, i], loc=0, scale=1))
        
    return np.array(tr_arr).T, scaler_01


def inverse_prob_int_transform(data: np.array, scaler) -> np.array:
    tr_arr = []
    for i in range(data.shape[1]):
        tr_arr.append(sts.norm.cdf(data[:, i]))

    tr_arr = np.array(tr_arr).T
    res = scaler.inverse_transform(tr_arr)
    
    return res


'''
DATE and TIME FEATURES
'''

def preprocessing_date(data: pd.DataFrame, date_feature: str) -> np.array:
    min_year = np.min(data[date_feature].apply(lambda x: x.year))
    max_year = np.max(data[date_feature].apply(lambda x: x.year))

    date_transformations = data[date_feature].apply(lambda x: np.array([np.cos(2*np.pi * x.day / 30),\
                                                                 np.sin(2*np.pi * x.day / 30),\
                                          np.cos(2*np.pi * x.month / 12), np.sin(2*np.pi * x.month / 12),\
                                          (x.year - min_year)/(max_year - min_year + 1e-7)])).values
    
    date_transformations = np.vstack(date_transformations)
    # date_transformations = date_transformations[:,:-1] #временно пока не придумаем что делать с годом

    return date_transformations


def preprocessing_time(data: pd.DataFrame, time_feature: str) -> np.array:

    time_transformations = data[time_feature].apply(lambda x: np.array([np.cos(2*np.pi * x.hour / 24),\
                                                                        np.sin(2*np.pi * x.hour / 24),\
                                                                        np.cos(2*np.pi * x.minute / 60),\
                                                                        np.sin(2*np.pi * x.minute / 60),\
                                                                        np.cos(2*np.pi * x.second / 60),\
                                                                        np.sin(2*np.pi * x.second / 60)])).values
    
    time_transformations = np.vstack(time_transformations)

    return time_transformations


# def convert_to_seconds(time):
#     return time.dt.hour * 60*60 + time.dt.minute * 60 + time.dt.second


def log1p_transform(data: pd.DataFrame, cols: list) -> pd.DataFrame:
    data = copy.deepcopy(data)
    data[cols] = np.log1p(data[cols])
    
    return data

def inverse_log1p_transform(data: pd.DataFrame, cols: list) -> pd.DataFrame:
    data = copy.deepcopy(data)
    data[cols] = np.exp(data[cols]) - 1
    
    return data