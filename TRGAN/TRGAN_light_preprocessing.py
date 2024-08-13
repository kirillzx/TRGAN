import pandas as pd
import numpy as np
import copy
import random

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
from rdt.transformers.categorical import FrequencyEncoder
from typing import Union, TypeVar



'''
CATEGORICAL FEATURES
'''

#forward transformation

def create_categorical_embeddings(data_init: pd.DataFrame, feat_names: list) -> Union[np.array, None, list]:
    freq_enc = []
    new_features = []
    data = copy.deepcopy(data_init)

    for i in range(len(feat_names)):
        enc = FrequencyEncoder()
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
    # synth_data_scaled_cl = synth_data_scaled_cl.astype(int)
    # synth_data_scaled_cl = np.where(synth_data_scaled_cl < 0, 0, synth_data_scaled_cl)

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