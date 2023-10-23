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
from scipy import signal
from functorch import vmap

from TRGAN.encoders import *


'''
CATEGORICAL FEATURES
'''

def onehot_emb_categorical(data: pd.DataFrame, cat_features):
    data_cat_onehot = pd.get_dummies(data[cat_features], columns=cat_features)
    return data_cat_onehot

def create_categorical_embeddings(data_cat_onehot: pd.DataFrame, dim_Xoh, lr=1e-3, epochs=20, batch_size=2**8, device='cpu', eps=2):
    data_dim_onehot = len(data_cat_onehot.columns)

    encoder_onehot = Encoder_onehot(data_dim_onehot, dim_Xoh).to(device)
    decoder_onehot = Decoder_onehot(dim_Xoh, data_dim_onehot).to(device)

    optimizer_Enc = optim.Adam(encoder_onehot.parameters(), lr=lr)
    optimizer_Dec = optim.Adam(decoder_onehot.parameters(), lr=lr)

    scheduler_Enc = torch.optim.lr_scheduler.ExponentialLR(optimizer_Enc, gamma=0.98)
    scheduler_Dec = torch.optim.lr_scheduler.ExponentialLR(optimizer_Dec, gamma=0.98)

    loader_onehot = DataLoader(torch.FloatTensor(data_cat_onehot.values), batch_size, shuffle=True)

    epochs = tqdm(range(epochs))

    #privacy parameters
    q = batch_size / len(data_cat_onehot)
    alpha = 2
    delta = 0.1 #or 1/batch_size
    n_iter = len(data_cat_onehot) // batch_size
    sensitivity = 37/batch_size
    std = np.sqrt((4 * q * alpha**2 * (sensitivity)**2 * np.sqrt(2 * n_iter * np.log(1/delta))) / (2 * (alpha-1) * eps))

    print(f'E_oh with {eps, delta}-Differential Privacy')

    for epoch in epochs:
        for batch_idx, X in enumerate(loader_onehot):
            loss = torch.nn.BCELoss()
            # loss = torch.nn.BCEWithLogitsLoss()
            
            H = encoder_onehot(X.float().to(device))
            X_tilde = decoder_onehot(H.to(device))

            criterion0 = (loss(X_tilde, X.to(device))).to(device)
            criterion = (criterion0 + np.random.normal(0, std)).to(device)
            
            optimizer_Enc.zero_grad()
            optimizer_Dec.zero_grad()
            
            criterion.backward()
            
            optimizer_Enc.step()
            optimizer_Dec.step()

        scheduler_Enc.step()
        scheduler_Dec.step()

        # print(f'epoch {epoch}: Loss = {criterion:.8f}')
        epochs.set_description(f'Loss E_oh: {criterion0.item()} || Privacy Loss E_oh: {criterion.item()}')

    encoder_onehot.eval()
    decoder_onehot.eval()

    data_cat_encode = encoder_onehot(torch.FloatTensor(data_cat_onehot.values).to(device)).detach().cpu().numpy()

    return data_cat_encode, encoder_onehot, decoder_onehot


'''
CONTINUOUS FEATURES
'''

def preprocessing_cont(X, cont_features, type_scale='Autoencoder', max_clusters=15,\
                    weight_threshold=0.005, epochs=20, lr=0.001, bs=2**8, dim_cont_emb=3, device='cpu', eps=0.01):
    data = copy.deepcopy(X)

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

    elif type_scale == 'Autoencoder':
        scaler = []
        
        scaler_cont = GaussianNormalizer(enforce_min_max_values=True, learn_rounding_scheme=True)
        scaler_cont.reset_randomization()
        data[cont_features] = scaler_cont.fit_transform(data[cont_features], column=cont_features)

        scaler_cont2 = MinMaxScaler((-1, 1))
        data[cont_features] = scaler_cont2.fit_transform(data[cont_features])

        encoder_cont_emb = Encoder_cont_emb(len(cont_features), dim_cont_emb).to(device)
        decoder_cont_emb = Decoder_cont_emb(dim_cont_emb, len(cont_features)).to(device)

        optimizer_Enc_cont_emb = optim.Adam(encoder_cont_emb.parameters(), lr, betas=(0.9, 0.999), amsgrad=True)
        optimizer_Dec_cont_emb = optim.Adam(decoder_cont_emb.parameters(), lr, betas=(0.9, 0.999), amsgrad=True)

        scheduler_Enc = torch.optim.lr_scheduler.ExponentialLR(optimizer_Enc_cont_emb, gamma=0.98)
        scheduler_Dec = torch.optim.lr_scheduler.ExponentialLR(optimizer_Dec_cont_emb, gamma=0.98)

        loader_cont_emb = DataLoader(torch.FloatTensor(data[cont_features].values), bs, shuffle=True)

        epochs = tqdm(range(epochs))

        #privacy parameters
        q = bs / len(data)
        alpha = 2
        delta = 0.1 #or 1/batch_size
        n_iter = len(data) // bs
        sensitivity = 4/bs
        std = np.sqrt((4 * q * alpha**2 * (sensitivity)**2 * np.sqrt(2 * n_iter * np.log(1/delta))) / (2 * (alpha-1) * eps))

        print(f'E_cont with {eps, delta}-Differential Privacy')

        for epoch in epochs:
            for batch_idx, X in enumerate(loader_cont_emb):
                loss = torch.nn.MSELoss()

                H = encoder_cont_emb(X.float().to(device))
                X_tilde = decoder_cont_emb(H.to(device))
                
                loss_mse = loss(X.float().to(device), X_tilde).to(device)
                criterion = (loss_mse + np.random.normal(0, std)).to(device)
                
                optimizer_Enc_cont_emb.zero_grad()
                optimizer_Dec_cont_emb.zero_grad()
                
                criterion.backward()
                
                optimizer_Enc_cont_emb.step()
                optimizer_Dec_cont_emb.step()

            scheduler_Enc.step()
            scheduler_Dec.step()

            epochs.set_description(f'Loss E_cont: {loss_mse.item()} || Privacy Loss E_cont: {criterion.item()}')

        encoder_cont_emb.eval()
        decoder_cont_emb.eval()

        X_cont = encoder_cont_emb(torch.FloatTensor(data[cont_features].values).to(device)).detach().cpu().numpy()

        scaler.append(decoder_cont_emb)
        scaler.append(scaler_cont2)
        scaler.append(scaler_cont)
        scaler.append(encoder_cont_emb)

    else:
        print('Choose preprocessing scheme for continuous features. Available: CBNormalize and Standardize')

    # return data[cont_features].values, scaler
    scaler = np.array(scaler, dtype=object)
    
    return X_cont, scaler


'''
DATE FEATURES
'''

def preprocessing_date(data: pd.DataFrame, date_feature):
    min_year = np.min(data[date_feature].apply(lambda x: x.year))
    max_year = np.max(data[date_feature].apply(lambda x: x.year))

    date_transformations = data[date_feature].apply(lambda x: np.array([np.cos(2*np.pi * x.day / 365),\
                                                                 np.sin(2*np.pi * x.day / 365),\
                                          np.cos(2*np.pi * x.month / 12), np.sin(2*np.pi * x.month / 12),\
                                          (x.year - min_year)/(max_year - min_year + 1e-7)])).values
    
    date_transformations = np.vstack(date_transformations)
    date_transformations = date_transformations[:,:-1] #временно пока не придумаем что делать с годом

    return date_transformations

'''
CLIENT FEATURES
'''

def create_client_embeddings(data: pd.DataFrame, client_info, dim_X_cl=4, lr=1e-3, epochs=20, batch_size=2**8, device='cpu', eps=0.5):
    label_encoders_array = []
    client_info_new_features = []

    for i in range(len(client_info)):
        enc = FrequencyEncoder()
        customer_enc = enc.fit_transform(data, column=client_info[i])[client_info[i]].values
        # customer_enc = enc.fit_transform(data[client_info[i]])
        
        label_encoders_array.append(enc)
        client_info_new_features.append(customer_enc)

    # cum_count_clients = data.groupby('customer').cumcount()
    # client_info_for_emb = np.concatenate([np.array(client_info_new_features).T, date_transformations[:,:-1], cum_count_clients.values.reshape(-1, 1)], axis=1)

    client_info_for_emb = np.array(client_info_new_features).T
    client_info_for_emb = client_info_for_emb.astype(float)



    encoder_cl_emb = Encoder_client_emb(client_info_for_emb.shape[1], dim_X_cl).to(device)
    decoder_cl_emb = Decoder_client_emb(dim_X_cl, client_info_for_emb.shape[1]).to(device)

    optimizer_Enc_cl_emb = optim.Adam(encoder_cl_emb.parameters(), lr)
    optimizer_Dec_cl_emb = optim.Adam(decoder_cl_emb.parameters(), lr)

    scaler_minmax_cl_emb = MinMaxScaler((-1, 1))
    client_info_for_emb = scaler_minmax_cl_emb.fit_transform(client_info_for_emb)

    loader_cl_emb = DataLoader(torch.FloatTensor(client_info_for_emb), batch_size, shuffle=True)

    epochs = tqdm(range(epochs))

    #privacy parameters
    q = batch_size / len(data)
    alpha = 2
    delta = 0.1 #or 1/batch_size
    n_iter = len(data) // batch_size
    sensitivity = 4/batch_size
    std = np.sqrt((4 * q * alpha**2 * (sensitivity)**2 * np.sqrt(2 * n_iter * np.log(1/delta))) / (2 * (alpha-1) * eps))

    print(f'E_cl with {eps, delta}-Differential Privacy')

    for epoch in epochs:
        for batch_idx, X in enumerate(loader_cl_emb):
            loss = torch.nn.MSELoss()

            H = encoder_cl_emb(X.float().to(device))
            X_tilde = decoder_cl_emb(H.to(device))
            
            loss_mse = loss(X.float().to(device), X_tilde).to(device)
            criterion = (loss_mse + np.random.normal(0, std)).to(device)
            
            optimizer_Enc_cl_emb.zero_grad()
            optimizer_Dec_cl_emb.zero_grad()
            
            criterion.backward()
            
            optimizer_Enc_cl_emb.step()
            optimizer_Dec_cl_emb.step()

        epochs.set_description(f'Loss E_cl: {loss_mse.item()} || Privacy Loss E_cl: {criterion.item()}')

    client_encoding = encoder_cl_emb(torch.FloatTensor(client_info_for_emb).to(device)).detach().cpu().numpy()

    return client_encoding, encoder_cl_emb, decoder_cl_emb, scaler_minmax_cl_emb, label_encoders_array

'''
CREATE EMBEDDINGS AND CONDITIONAL VECTOR
'''

def create_embeddings(X_cont, X_oh_emb, X_cl):
    data_transformed = np.concatenate([X_cont, X_oh_emb], axis=1)
    scaler = MinMaxScaler((-1, 1))

    data_transformed = scaler.fit_transform(data_transformed)

    data_transformed = np.concatenate([X_cl, data_transformed], axis=1)

    return data_transformed, scaler

def behaviour_encoding(data, dim, name_client_id='customer',  name_agg_feature='amount'): 
    quantiles_array = []
    for i in range(1, dim+1):
        quant_object = data.groupby(name_client_id)[name_agg_feature].expanding().apply(lambda x: np.quantile(np.abs(np.fft.fft(x)), i/(dim+1)))

        idx_quant = list(map(lambda x: x[1], quant_object.index.values))
        quantiles = np.array(list(map(lambda x: x[0], sorted(list(zip(quant_object.values, idx_quant)), key=lambda x: x[1]))))

        quantiles_array.append(quantiles)

    quantiles_array = np.array(quantiles_array).T
    # quantiles_array_f = np.apply_along_axis(lambda x: np.abs(np.fft.fft(x))[1:dim//2], 1, quantiles_array)
    
    return np.log1p(quantiles_array)

def create_cond_vector(data, X_emb, date_feature, time, dim_Vc_h, dim_q, name_client_id,\
                    name_agg_feature, lr=1e-3, epochs=20, batch_size = 2**8, model_time='poisson',\
                    n_splits=2, opt_time=True, xi_array=[], q_array=[], device='cpu', eps=0.5):

    if time == 'synth':
        data_synth_time, deltas_by_clients, synth_deltas_by_clients, xiP_array, idx_array = generate_synth_time(data,\
                                        name_client_id, date_feature[0], model_time, n_splits, opt_time, xi_array, q_array)
        date_transformations = preprocessing_date(data_synth_time, date_feature[0])

    elif time == 'initial':
        data_synth_time = data[date_feature]
        date_transformations = preprocessing_date(data, date_feature[0])
        deltas_by_clients = 'Only when time="synth"'
        synth_deltas_by_clients = 'Only when time="synth"'
        xiP_array = []
        idx_array = []

    else:
        print('Choose time generation type')

    # behaviour_cl_enc = behaviour_encoding(data.reset_index(), dim_q, name_client_id, name_agg_feature)
    # scaler_behaviour_cl_enc = MinMaxScaler()
    # behaviour_cl_enc = scaler_behaviour_cl_enc.fit_transform(behaviour_cl_enc)
    
    behaviour_cl_enc = ['', '']


    hidden_dim = dim_Vc_h
    data_dim = len(X_emb[0])

    encoder = Encoder(data_dim, hidden_dim).to(device)
    decoder = Decoder(hidden_dim, data_dim).to(device)

    optimizer_Enc = optim.Adam(encoder.parameters(), lr)
    optimizer_Dec = optim.Adam(decoder.parameters(), lr)

    #privacy parameters
    q = batch_size / len(X_emb)
    alpha = 2
    delta = 0.1 #or 1/batch_size
    n_iter = len(X_emb) // batch_size
    sensitivity = 4/batch_size
    std = np.sqrt((4 * q * alpha**2 * (sensitivity)**2 * np.sqrt(2 * n_iter * np.log(1/delta))) / (2 * (alpha-1) * eps))

    print(f'E_cv with {eps, delta}-Differential Privacy')

    loader = DataLoader(torch.FloatTensor(X_emb), batch_size=batch_size, shuffle=True)

    epochs = tqdm(range(epochs))

    for epoch in epochs:
        for batch_idx, X in enumerate(loader):
            loss = torch.nn.MSELoss()

            H = encoder(X.float().to(device))
            X_tilde = decoder(H.to(device))
            
            loss_mse = loss(X.float().to(device), X_tilde).to(device)
            # criterion = loss_mse + np.random.laplace(loc=0, scale=4*c**2 / epsilon)
            criterion = (loss_mse + np.random.normal(0, std)).to(device)
            
            optimizer_Enc.zero_grad()
            optimizer_Dec.zero_grad()
            
            criterion.backward()
        
            optimizer_Enc.step()
            optimizer_Dec.step()

        epochs.set_description('Loss E_cv: %.9f || Privacy Loss E_cv: %.9f' % (loss_mse.item(), criterion.item()))

    data_encode = encoder(torch.FloatTensor(X_emb).to(device)).detach().cpu().numpy()
    
    # cond_vector = np.concatenate([data_encode, date_transformations, behaviour_cl_enc], axis=1)
    cond_vector = np.concatenate([data_encode, date_transformations], axis=1)

    return cond_vector, data_synth_time, date_transformations, behaviour_cl_enc, encoder,\
            deltas_by_clients, synth_deltas_by_clients, xiP_array, idx_array 



'''
GENERATOR
'''
def init_weight_seq(layer):
    if type(layer) == nn.Linear:
        torch.nn.init.normal_(layer.weight, 0, 0.02)

def convolve_vec(tensor, filter):
    # return torch.nn.functional.conv1d(tensor.view(1, 1, -1), filter.to(device).view(1, 1, -1), padding='same').view(-1)
    return torch.nn.functional.conv1d(tensor.detach().cpu().view(1, 1, -1), filter.view(1, 1, -1), padding='same').view(-1)

conv_vec = vmap(convolve_vec)
# gauss_filter_dim = 25

class Generator(nn.Module):
    def __init__(self, z_dim, data_dim, h_dim, num_blocks, gauss_filter_dim, device):
        super(Generator, self).__init__()
        
        self.z_dim = z_dim
        self.data_dim = data_dim
        self.h_dim = h_dim
        self.num_blocks = num_blocks
        self.gauss_filter_dim = gauss_filter_dim
        self.device = device

        self.fc1 = nn.Linear(self.z_dim, self.h_dim)
        self.relu = nn.LeakyReLU(0.2)
        self.lrelu = nn.LeakyReLU(0.1)
        # self.relu = nn.ELU(0.9)
        # self.relu = nn.PReLU()
        
        self.fc2 = nn.Linear(self.h_dim, self.data_dim)
        self.tanh = nn.Tanh()
        self.layernorm0 = nn.LayerNorm(self.h_dim)
        # self.layernorm0 = nn.BatchNorm1d(self.h_dim)
  

        self.linear_layers = nn.ModuleList([nn.Linear(3 * 2**6, self.h_dim) for _ in range(self.num_blocks)])
        self.linear_layers_conv1 = nn.ModuleList([nn.Linear(self.h_dim, 2**6) for _ in range(self.num_blocks)])
        self.linear_layers_conv2 = nn.ModuleList([nn.Linear(self.h_dim, 2**6) for _ in range(self.num_blocks)])
        self.linear_layers_conv3 = nn.ModuleList([nn.Linear(self.h_dim, 2**6) for _ in range(self.num_blocks)])

        self.feed_forward_generator_layers = nn.ModuleList(
            [nn.Sequential(
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.h_dim, self.h_dim)
        ) for _ in range(self.num_blocks)]
        )

        self.feed_forward_generator_layers2 = nn.ModuleList(
            [nn.Sequential(
            nn.Linear(self.h_dim, self.h_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(self.h_dim, self.h_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
        ) for _ in range(self.num_blocks)]
        )

        self.layernorm_layers_1 = nn.ModuleList([nn.LayerNorm(self.h_dim) for _ in range(self.num_blocks)])
        self.layernorm_layers_2 = nn.ModuleList([nn.LayerNorm(self.h_dim) for _ in range(self.num_blocks)])
        # self.layernorm_layers_1 = nn.ModuleList([nn.BatchNorm1d(self.h_dim) for _ in range(self.num_blocks)])
        # self.layernorm_layers_2 = nn.ModuleList([nn.BatchNorm1d(self.h_dim) for _ in range(self.num_blocks)])

        self.filter1 = signal.windows.gaussian(gauss_filter_dim, 0.5)
        self.filter2 = signal.windows.gaussian(gauss_filter_dim, 1)
        self.filter3 = signal.windows.gaussian(gauss_filter_dim, 3)

        # self.init_weights()
        # self.linear_layers.apply(init_weight_seq)
        # self.linear_layers_conv1.apply(init_weight_seq)
        # self.linear_layers_conv2.apply(init_weight_seq)
        # self.linear_layers_conv3.apply(init_weight_seq)
        
    def init_weights(self):
        torch.nn.init.normal_(self.fc1.weight, 0, 0.02)
        torch.nn.init.normal_(self.fc2.weight, 0, 0.02)

            
    def forward(self, x):
        x_size = x.size()
        out = self.layernorm0(self.relu(self.fc1(x)))
        
        for i in range(self.num_blocks):
            res = out

            # x1 = conv_vec(out, torch.FloatTensor(self.filter1).expand(x_size[0], self.gauss_filter_dim)).to(self.device)
            # x2 = conv_vec(out, torch.FloatTensor(self.filter2).expand(x_size[0], self.gauss_filter_dim)).to(self.device)
            # x3 = conv_vec(out, torch.FloatTensor(self.filter3).expand(x_size[0], self.gauss_filter_dim)).to(self.device)

            # x1 = self.lrelu(self.linear_layers_conv1[i](x1))
            # x2 = self.lrelu(self.linear_layers_conv2[i](x2))
            # x3 = self.lrelu(self.linear_layers_conv3[i](x3))

            # out = torch.cat([x1, x2, x3], dim=1)
 
            # out = self.linear_layers[i](out)

            out = self.feed_forward_generator_layers2[i](out)

            #add & norm
            out += res
            out = self.layernorm_layers_1[i](out)

            # #feed forward
            # res = out
            # out = self.feed_forward_generator_layers[i](out)
            # # #add & norm
            # out += res
            # out = self.layernorm_layers_2[i](out)

        out = self.fc2(out)
        return self.tanh(out)


class Supervisor(nn.Module):
    def __init__(self, z_dim, data_dim, h_dim, num_blocks, gauss_filter_dim, device):
        super(Supervisor, self).__init__()
        
        self.z_dim = z_dim
        self.data_dim = data_dim
        self.h_dim = h_dim
        self.num_blocks = num_blocks
        self.gauss_filter_dim = gauss_filter_dim
        self.device = device

        self.fc1 = nn.Linear(self.z_dim, self.h_dim)
        self.relu = nn.LeakyReLU(0.2)
        self.lrelu = nn.LeakyReLU(0.1)
        # self.relu = nn.ELU(0.9)
        # self.relu = nn.PReLU()
        
        self.fc2 = nn.Linear(self.h_dim, self.data_dim)
        self.tanh = nn.Tanh()

        self.layernorm0 = nn.LayerNorm(self.h_dim)
        # self.layernorm0 = nn.BatchNorm1d(self.h_dim)

        # self.linear_layers = nn.ModuleList([nn.Linear(self.h_dim, self.h_dim) for _ in range(self.num_blocks)])
        self.linear_layers = nn.ModuleList([nn.Linear(3 * 2**6, self.h_dim) for _ in range(self.num_blocks)])
        self.linear_layers_conv1 = nn.ModuleList([nn.Linear(self.h_dim, 2**6) for _ in range(self.num_blocks)])
        self.linear_layers_conv2 = nn.ModuleList([nn.Linear(self.h_dim, 2**6) for _ in range(self.num_blocks)])
        self.linear_layers_conv3 = nn.ModuleList([nn.Linear(self.h_dim, 2**6) for _ in range(self.num_blocks)])

        self.feed_forward_generator_layers = nn.ModuleList(
            [nn.Sequential(
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.h_dim, self.h_dim)
        ) for _ in range(self.num_blocks)]
        )

        self.feed_forward_generator_layers2 = nn.ModuleList(
            [nn.Sequential(
            nn.Linear(self.h_dim, self.h_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(self.h_dim, self.h_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
        ) for _ in range(self.num_blocks)]
        )

        self.layernorm_layers_1 = nn.ModuleList([nn.LayerNorm(self.h_dim) for _ in range(self.num_blocks)])
        self.layernorm_layers_2 = nn.ModuleList([nn.LayerNorm(self.h_dim) for _ in range(self.num_blocks)])
        # self.layernorm_layers_1 = nn.ModuleList([nn.BatchNorm1d(self.h_dim) for _ in range(self.num_blocks)])
        # self.layernorm_layers_2 = nn.ModuleList([nn.BatchNorm1d(self.h_dim) for _ in range(self.num_blocks)])

        self.filter1 = signal.windows.gaussian(gauss_filter_dim, 0.5)
        self.filter2 = signal.windows.gaussian(gauss_filter_dim, 1)
        self.filter3 = signal.windows.gaussian(gauss_filter_dim, 3)

        # self.init_weights()
        # self.linear_layers.apply(init_weight_seq)
        # self.linear_layers_conv1.apply(init_weight_seq)
        # self.linear_layers_conv2.apply(init_weight_seq)
        # self.linear_layers_conv3.apply(init_weight_seq)

    def init_weights(self):
        torch.nn.init.normal_(self.fc1.weight, 0, 0.02)
        torch.nn.init.normal_(self.fc2.weight, 0, 0.02)

            
    def forward(self, x):
        x_size = x.size()
        out = self.layernorm0(self.relu(self.fc1(x)))
        
        for i in range(self.num_blocks):
            res = out

            # x1 = conv_vec(out, torch.FloatTensor(self.filter1).expand(x_size[0], self.gauss_filter_dim)).to(self.device)
            # x2 = conv_vec(out, torch.FloatTensor(self.filter2).expand(x_size[0], self.gauss_filter_dim)).to(self.device)
            # x3 = conv_vec(out, torch.FloatTensor(self.filter3).expand(x_size[0], self.gauss_filter_dim)).to(self.device)

            
            # x1 = self.lrelu(self.linear_layers_conv1[i](x1))
            # x2 = self.lrelu(self.linear_layers_conv2[i](x2))
            # x3 = self.lrelu(self.linear_layers_conv3[i](x3))

           
            # out = torch.cat([x1, x2, x3], dim=1)
 
            # out = self.linear_layers[i](out)

            out = self.feed_forward_generator_layers2[i](out)

            #add & norm
            out += res
            out = self.layernorm_layers_1[i](out)

            # #feed forward
            # res = out
            # out = self.feed_forward_generator_layers[i](out)
            # # #add & norm
            # out += res
            # out = self.layernorm_layers_2[i](out)

        out = self.fc2(out)
        return self.tanh(out)

class Discriminator(nn.Module):
    def __init__(self, data_dim, h_dim, num_blocks, gauss_filter_dim, device):
        super(Discriminator, self).__init__()
        
        self.data_dim = data_dim
        self.h_dim = h_dim
        self.num_blocks = num_blocks
        self.gauss_filter_dim = gauss_filter_dim
        self.device = device

        self.fc1 = nn.Linear(self.data_dim, self.h_dim)
        self.lrelu = nn.LeakyReLU(0.1)
        # self.relu = nn.ELU(0.9)
        self.relu = nn.ReLU()

        self.layernorm0 = nn.LayerNorm(self.h_dim)
        # self.layernorm0 = nn.BatchNorm1d(self.h_dim)

        # self.linear_layers = nn.ModuleList([nn.Linear(self.h_dim, self.h_dim) for _ in range(self.num_blocks)])
        self.linear_layers = nn.ModuleList([nn.Linear(3 * 2**6, self.h_dim) for _ in range(self.num_blocks)])
        self.linear_layers_conv1 = nn.ModuleList([nn.Linear(self.h_dim, 2**6) for _ in range(self.num_blocks)])
        self.linear_layers_conv2 = nn.ModuleList([nn.Linear(self.h_dim, 2**6) for _ in range(self.num_blocks)])
        self.linear_layers_conv3 = nn.ModuleList([nn.Linear(self.h_dim, 2**6) for _ in range(self.num_blocks)])

        # self.feed_forward_discriminator_layers = nn.ModuleList(
        #     [nn.Sequential(
        #     nn.Linear(self.h_dim, self.h_dim),
        #     nn.ReLU(),
        #     # nn.Dropout(0.15),
        #     nn.Linear(self.h_dim, self.h_dim)
        # ) for _ in range(self.num_blocks)]
        # )

        self.feed_forward_discriminator_layers2 = nn.ModuleList(
            [nn.Sequential(
            nn.Linear(self.h_dim, self.h_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(self.h_dim, self.h_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
        ) for _ in range(self.num_blocks)]
        )

        self.layernorm_layers_1 = nn.ModuleList([nn.LayerNorm(self.h_dim) for _ in range(self.num_blocks)])
        self.layernorm_layers_2 = nn.ModuleList([nn.LayerNorm(self.h_dim) for _ in range(self.num_blocks)])
        # self.layernorm_layers_1 = nn.ModuleList([nn.BatchNorm1d(self.h_dim) for _ in range(self.num_blocks)])
        # self.layernorm_layers_2 = nn.ModuleList([nn.BatchNorm1d(self.h_dim) for _ in range(self.num_blocks)])

        self.filter1 = signal.windows.gaussian(gauss_filter_dim, 0.5)
        self.filter2 = signal.windows.gaussian(gauss_filter_dim, 1)
        self.filter3 = signal.windows.gaussian(gauss_filter_dim, 3)

        self.fc2 = nn.Linear(self.h_dim, 1)
        self.sigmoid = nn.Sigmoid()

        # self.init_weights()
        # self.linear_layers.apply(init_weight_seq)
        # self.linear_layers_conv1.apply(init_weight_seq)
        # self.linear_layers_conv2.apply(init_weight_seq)
        # self.linear_layers_conv3.apply(init_weight_seq)

    def init_weights(self):
        torch.nn.init.normal_(self.fc1.weight, 0, 0.02)
        torch.nn.init.normal_(self.fc2.weight, 0, 0.02)
    
            
    def forward(self, x):
        x_size = x.size()
        out = self.layernorm0(self.relu(self.fc1(x)))

        for i in range(self.num_blocks):
            res = out

            # x1 = conv_vec(out, torch.FloatTensor(self.filter1).expand(x_size[0], self.gauss_filter_dim)).to(self.device)
            # x2 = conv_vec(out, torch.FloatTensor(self.filter2).expand(x_size[0], self.gauss_filter_dim)).to(self.device)
            # x3 = conv_vec(out, torch.FloatTensor(self.filter3).expand(x_size[0], self.gauss_filter_dim)).to(self.device)

            # x1 = self.lrelu(self.linear_layers_conv1[i](x1))
            # x2 = self.lrelu(self.linear_layers_conv2[i](x2))
            # x3 = self.lrelu(self.linear_layers_conv3[i](x3))

            # out = torch.cat([x1, x2, x3], dim=1)

            # out = self.linear_layers[i](out)

            out = self.feed_forward_discriminator_layers2[i](out)

            #add & norm
            out += res
            out = self.layernorm_layers_1[i](out)

            # # #feed forward
            # res = out
            # out = self.feed_forward_discriminator_layers[i](out)
            
            # # #add & norm
            # out += res
            # out = self.layernorm_layers_2[i](out)
  
        out = self.fc2(out)
        return out

def grad_penalty(discriminator, real_data, gen_data, device):
        batch_size = real_data.size()[0]
        t = torch.rand((batch_size, 1), requires_grad=True).to(device)
        t = t.expand_as(real_data)

        # mixed sample from real and fake; make approx of the 'true' gradient norm
        interpol = t * real_data + (1-t) * gen_data
        
        prob_interpol = discriminator(interpol).to(device)
        torch.autograd.set_detect_anomaly(True)
        gradients = grad(outputs=prob_interpol, inputs=interpol.to(device),
                               grad_outputs=torch.ones(prob_interpol.size()).to(device), create_graph=True, retain_graph=True)[0]
        gradients = gradients.view(batch_size, -1).to(device)
        #grad_norm = torch.norm(gradients, dim=1).mean()
        #self.losses['gradient_norm'].append(grad_norm.item())

        # add epsilon for stability
        eps = 1e-10
        gradients_norm = torch.sqrt(torch.sum(gradients**2, dim=1, dtype=torch.double) + eps)
        
        return 10*(torch.max(torch.zeros(1,dtype=torch.double).to(device), gradients_norm.mean() - 1) ** 2)

def train_generator(X_emb, cond_vector, dim_Vc, dim_X_emb, dim_noise=5, batch_size=2**9, lr_rates=[3e-4, 3e-4, 3e-4, 3e-4],\
                     num_epochs=15, num_blocks_gen=1, num_blocks_dis=2, h_dim=2**7, lambda1=3, alpha=0.75, window_size=25, device='cpu'):
    # date_transf_dim = num_date_features
    # pos_dim = behaviour_cl_enc.shape[1]
    data_dim = dim_X_emb
    z_dim = dim_noise + dim_Vc
    gauss_filter_dim = window_size

    generator = Generator(z_dim, data_dim, h_dim, num_blocks_gen, gauss_filter_dim, device).to(device)
    discriminator = Discriminator(data_dim + dim_Vc, h_dim, num_blocks_dis, gauss_filter_dim, device).to(device)
    supervisor = Supervisor(data_dim + dim_Vc, data_dim, h_dim, num_blocks_gen, gauss_filter_dim, device).to(device)
    discriminator2 = Discriminator(data_dim + dim_Vc, h_dim, num_blocks_dis, gauss_filter_dim, device).to(device)

    optimizer_G = optim.Adam(generator.parameters(), lr=lr_rates[0], betas=(0.9, 0.999), amsgrad=True)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr_rates[1], betas=(0.9, 0.999), amsgrad=True)
    optimizer_S = optim.Adam(supervisor.parameters(), lr=lr_rates[2], betas=(0.9, 0.999), amsgrad=True)
    optimizer_D2 = optim.Adam(discriminator2.parameters(), lr=lr_rates[3], betas=(0.9, 0.999), amsgrad=True)

    scheduler_G = torch.optim.lr_scheduler.ExponentialLR(optimizer_G, gamma=0.97)
    scheduler_D = torch.optim.lr_scheduler.ExponentialLR(optimizer_D, gamma=0.97)
    scheduler_S = torch.optim.lr_scheduler.ExponentialLR(optimizer_S, gamma=0.97)
    scheduler_D2 = torch.optim.lr_scheduler.ExponentialLR(optimizer_D2, gamma=0.97)

    data_with_cv = torch.cat([torch.FloatTensor(X_emb), torch.FloatTensor(cond_vector)], axis=1)

    idx_batch_array = np.arange(len(data_with_cv)//batch_size * batch_size)
    last_idx = np.setdiff1d(np.arange(len(data_with_cv)), idx_batch_array)
    split_idx = np.split(idx_batch_array, batch_size)
    split_idx_perm = np.random.permutation(split_idx)
    split_idx_perm = np.append(split_idx_perm, last_idx)

    loader_g = DataLoader(data_with_cv[split_idx_perm], batch_size=batch_size, shuffle=True)

    epochs = tqdm(range(num_epochs))
    loss_array = []

    b_d1 = 0.01
    b_d2 = 0.01

    for epoch in epochs:
        for batch_idx, X in enumerate(loader_g):
            loss = torch.nn.MSELoss()
            batch_size = X.size(0)

            Vc = X[:, -dim_Vc:].to(device)
            
            # noise = torch.randn(batch_size, dim_noise).to(device)
            noise = torch.FloatTensor(dclProcess(batch_size - 1, dim_noise)).to(device)
            z = torch.cat([noise, Vc], dim=1).to(device)
            
            fake = generator(z).detach()
            X = X.to(device)
            
            discriminator.trainable = True
            
            disc_loss = (-torch.mean(discriminator(X)) + torch.mean(discriminator(torch.cat([fake, Vc], dim=1)))).to(device)
     
            fake_super = supervisor(torch.cat([fake, Vc], dim=1)).to(device)
            disc2_loss = (-torch.mean(discriminator2(X)) + torch.mean(discriminator2(torch.cat([fake_super, Vc], dim=1)))).to(device) 

            optimizer_D.zero_grad()
            disc_loss.backward()
            optimizer_D.step()

            optimizer_D2.zero_grad()
            disc2_loss.backward()
            optimizer_D2.step()

            for dp in discriminator.parameters():
                        dp.data.clamp_(-b_d1, b_d1)

            for dp in discriminator2.parameters():
                        dp.data.clamp_(-b_d2, b_d2)

                    
            if batch_idx % 2 == 0:
                discriminator.trainable = False

                gen_loss1 = -torch.mean(discriminator(torch.cat([generator(z), Vc], dim=1))).to(device)
                supervisor_loss = (-torch.mean(discriminator2(torch.cat([supervisor(torch.cat([generator(z), Vc], dim=1).detach()), Vc], dim=1))) +\
                                    lambda1 * loss(supervisor(torch.cat([generator(z), Vc], dim=1).detach()), X[:,:-dim_Vc])).to(device)
                
                # gen_loss1 = torch.mean(torch.log(discriminator(torch.cat([generator(z), Vc], dim=1)))).to(device)
                # supervisor_loss = (torch.mean(torch.log(discriminator2(torch.cat([supervisor(torch.cat([generator(z), Vc], dim=1).detach()), Vc], dim=1)))) +\
                #                     lambda1 * loss(supervisor(torch.cat([generator(z), Vc], dim=1).detach()), X[:,:-dim_Vc])).to(device)
                
                gen_loss = (alpha * gen_loss1 + (1 - alpha) * supervisor_loss)
                
                
                supervisor_loss2 = ((-torch.mean(discriminator2(torch.cat([supervisor(torch.cat([generator(z), Vc], dim=1).detach()),\
                    Vc], dim=1)))) + lambda1 * loss(supervisor(torch.cat([generator(z), Vc], dim=1).detach()), X[:,:-dim_Vc])).to(device)
                
                # supervisor_loss2 = ((torch.mean(torch.log(discriminator2(torch.cat([supervisor(torch.cat([generator(z), Vc], dim=1).detach()),\
                #     Vc], dim=1))))) + lambda1 * loss(supervisor(torch.cat([generator(z), Vc], dim=1).detach()), X[:,:-dim_Vc])).to(device)

                optimizer_G.zero_grad()
                gen_loss.backward()
                optimizer_G.step()
                
                optimizer_S.zero_grad()
                supervisor_loss2.backward()
                optimizer_S.step()

        # scheduler_G.step()
        # scheduler_D.step()
        # scheduler_S.step()
        # scheduler_D2.step()

        epochs.set_description('Discriminator Loss: %.5f || Discriminator 2 Loss: %.5f || Generator Loss: %.5f || Supervisor Loss: %.5f' %\
            (disc_loss.item(), disc2_loss.item(), gen_loss.item(), supervisor_loss2.item()))
        loss_array.append([disc_loss.item(), disc2_loss.item(), gen_loss.item(), supervisor_loss2.item()])

    return generator, supervisor, loss_array, discriminator, discriminator2

def sample_cond_vector_with_time(n_samples, len_cond_vector, X_emb, data, behaviour_cl_enc, date_feature, name_client_id, time='synth',
                        model_time='poisson', n_splits=2, opt_time=True, xi_array=[], q_array=[]):
    cond_vector_array = []
    synth_time_array = []
    residual = n_samples%len_cond_vector

    if n_samples > len_cond_vector:

        for i in range(n_samples//len_cond_vector):
            cond_vector, synth_time, _, _, _, _, _ = create_cond_vector_with_time_gen(X_emb, data,\
                    behaviour_cl_enc, date_feature, name_client_id, time, model_time, n_splits, opt_time, xi_array, q_array)
            cond_vector_array.append(cond_vector)
            synth_time_array.append(synth_time)

        if residual != 0:
            cond_vector, synth_time, _, _, _, _, _ = create_cond_vector_with_time_gen(X_emb[:residual], data[:residual],\
                            behaviour_cl_enc[:residual], date_feature, name_client_id, time,\
                            model_time, n_splits, opt_time, xi_array, q_array)
            cond_vector_array.append(cond_vector)
            synth_time_array.append(synth_time)

        synth_time = pd.DataFrame(np.vstack(synth_time_array), columns=date_feature)
        cond_vector = np.vstack(cond_vector_array)

    else:
        cond_vector, synth_time, _, _, _, _, _ = create_cond_vector_with_time_gen(X_emb, data, behaviour_cl_enc, date_feature, name_client_id, time,
                            model_time, n_splits, opt_time, xi_array, q_array)

    return synth_time, cond_vector

def sample(n_samples, generator, supervisor, noise_dim, cond_vector, X_emb, encoder, data, behaviour_cl_enc,\
            date_feature, name_client_id, time='initial', model_time='poisson', n_splits=2, opt_time=True, xi_array=[], q_array=[], device='cpu'):
    if n_samples <= len(cond_vector):
        
        X_emb_cv = encoder(torch.FloatTensor(X_emb).to(device)).detach().cpu().numpy()
        synth_time, cond_vector = sample_cond_vector_with_time(n_samples, len(cond_vector), X_emb_cv, data,\
                        behaviour_cl_enc, date_feature, name_client_id, time, model_time, n_splits, opt_time, xi_array, q_array)
        
        # noise = torch.randn(n_samples, noise_dim)
        noise = torch.FloatTensor(dclProcess(n_samples - 1, noise_dim)).to(device)
        z = torch.cat([noise.to(device), torch.FloatTensor(cond_vector[:n_samples]).to(device)], axis=1).to(device)
        synth_data = supervisor(torch.cat([generator(z).detach(), torch.FloatTensor(cond_vector[:n_samples]).to(device)], dim=1)).detach().cpu().numpy()
        synth_time = synth_time[:n_samples]
    # synth_time = data[date_feature]

    else:
        X_emb = encoder(torch.FloatTensor(X_emb).to(device)).detach().cpu().numpy()
        synth_time, cond_vector = sample_cond_vector_with_time(n_samples, len(cond_vector), X_emb, data,\
                        behaviour_cl_enc, date_feature, name_client_id, time, model_time, n_splits, opt_time, xi_array, q_array)
        
        # noise = torch.randn(n_samples, noise_dim)
        noise = torch.FloatTensor(dclProcess(n_samples - 1, noise_dim)).to(device)
        z = torch.cat([noise.to(device), torch.FloatTensor(cond_vector).to(device)], axis=1).to(device)
        synth_data = supervisor(torch.cat([generator(z).detach(), torch.FloatTensor(cond_vector).to(device)], dim=1)).detach().cpu().numpy()

    synth_time = synth_time.sort_values(by='transaction_date')

    return synth_data, synth_time


def create_cond_vector_with_time_gen(X_emb, data, behaviour_cl_enc, date_feature, name_client_id, time='initial',
                        model_time='poisson', n_splits=3, opt_time=True, xi_array=[], q_array=[]):

    if time == 'synth':
        data_synth_time, deltas_by_clients, synth_deltas_by_clients, xiP_array, idx_array = generate_synth_time(data,\
                    name_client_id, date_feature[0], model_time, n_splits, opt_time, xi_array, q_array)
        date_transformations = preprocessing_date(data_synth_time, date_feature[0])

    elif time == 'initial':
        data_synth_time = data[date_feature]
        date_transformations = preprocessing_date(data, date_feature[0])
        deltas_by_clients = data.groupby(name_client_id)[date_feature[0]].apply(lambda x: (x - x.shift()).dt.days.values[1:])
        synth_deltas_by_clients = deltas_by_clients
        xiP_array = []
        idx_array = []

    else:
        print('Choose time generation type')
    
    # cond_vector = np.concatenate([X_emb, date_transformations, behaviour_cl_enc], axis=1)
    cond_vector = np.concatenate([X_emb, date_transformations], axis=1)

    return cond_vector, data_synth_time, date_transformations, deltas_by_clients, synth_deltas_by_clients, xiP_array, idx_array


def undummify(df, prefix_sep="_"):
    cols2collapse = {
        item.split(prefix_sep)[0]: (prefix_sep in item) for item in df.columns
    }
    series_list = []
    for col, needs_to_collapse in cols2collapse.items():
        if needs_to_collapse:
            undummified = (
                df.filter(like=col)
                .idxmax(axis=1)
                .apply(lambda x: x.split(prefix_sep, maxsplit=1)[1])
                .rename(col)
            )
            series_list.append(undummified)
        else:
            series_list.append(df[col])
    undummified_df = pd.concat(series_list, axis=1)
    return undummified_df

def inverse_transforms(n_samples, synth_data, synth_time, client_info, cont_features, X_oh, scaler_emb, scaler_cl_emb,\
        scaler_cont, label_encoders, decoder_cl_emb, decoder_onehot, dim_Xcl, dim_X_cont, type_scale_cont='CBNormalize', device='cpu'):
    
    synth_data_scaled = scaler_emb.inverse_transform(synth_data[:, dim_Xcl:])

    '''
    CLIENTS FEATURES
    '''
    # client inverse transforms
    synth_data_scaled_cl = scaler_cl_emb.inverse_transform(decoder_cl_emb(torch.FloatTensor(synth_data[:, :dim_Xcl]).to(device)).detach().cpu().numpy())
    # synth_data_scaled_cl = synth_data_scaled_cl.astype(int)
    # synth_data_scaled_cl = np.where(synth_data_scaled_cl < 0, 0, synth_data_scaled_cl)
  
    customer_dec_array = []
    for i in range(len(client_info)):
        # customer_dec = label_encoders[i].inverse_transform(synth_data_scaled_cl[:, i])
        customer_dec = label_encoders[i].reverse_transform(pd.DataFrame(synth_data_scaled_cl[:, i], columns=[client_info[i]]))
        customer_dec_array.append(customer_dec.values)

    customer_dec_array = np.array(customer_dec_array)
    synth_data_scaled_cl = customer_dec_array.T[0]

    '''
    CONTINUOUS FEATURES
    '''
    if type_scale_cont == 'CBNormalize':
        synth_cont_feat = synth_data_scaled[:,:dim_X_cont]

        # if n_samples > len(scaler_cont[len(scaler_cont)//2]):
        if n_samples > len(scaler_cont[-1]):
            cont_synth = []
            # for i in range(len(scaler_cont)//2):
            for i in range(len(scaler_cont)//2):
                additional_components = np.random.choice(scaler_cont[len(scaler_cont)//2 + i].T[0], n_samples - len(scaler_cont[len(scaler_cont)//2]))
                new_components = np.concatenate([scaler_cont[len(scaler_cont)//2 + i].T[0], np.array(additional_components)])
                
                cont_synth.append(scaler_cont[i].reverse_transform(pd.DataFrame(np.concatenate([synth_cont_feat[:, i].reshape(-1, 1),\
                                                    new_components.reshape(-1, 1)], axis=1), \
                                                    columns=[cont_features[i]+'.normalized', cont_features[i]+'.component'])))
                
            synth_cont_feat = np.hstack(cont_synth)

        else:    
            cont_synth = []
            for i in range(len(scaler_cont)//2):
                cont_synth.append(scaler_cont[i].reverse_transform(pd.DataFrame(np.concatenate([synth_cont_feat[:, i].reshape(-1, 1),\
                                                    scaler_cont[len(scaler_cont)//2 + i][:n_samples].reshape(-1, 1)], axis=1), \
                                                    columns=[cont_features[i]+'.normalized', cont_features[i]+'.component'])))

            synth_cont_feat = np.hstack(cont_synth)

    elif type_scale_cont == 'Standardize':
        synth_cont_feat = synth_data_scaled[:,:dim_X_cont]
        synth_cont_feat = scaler_cont[0].inverse_transform(synth_cont_feat)

    elif type_scale_cont == 'Autoencoder':
        synth_cont_feat = synth_data_scaled[:,:dim_X_cont]
        synth_cont_feat = (scaler_cont[0](torch.FloatTensor(synth_cont_feat).to(device))).detach().cpu().numpy()
        synth_cont_feat = scaler_cont[1].inverse_transform(synth_cont_feat)
        scaler_cont[2].reset_randomization()
        synth_cont_feat = scaler_cont[2].reverse_transform(pd.DataFrame(synth_cont_feat, columns=cont_features))

    else:
        print('Incorrect preprocessing type for continuous features')


    '''
    CATEGORICAL FEATURES
    '''
    synth_cat_feat = synth_data_scaled[:, dim_X_cont:]
    synth_cat_decoded = np.abs(np.around(decoder_onehot(torch.FloatTensor(synth_cat_feat).to(device)).detach().cpu().numpy())).astype(int)
    synth_df_cat = pd.DataFrame(synth_cat_decoded, columns=X_oh.columns)
    synth_df_cat_feat_undum = undummify(synth_df_cat, prefix_sep="_")
    synth_df_cat_feat_undum['mcc'] = synth_df_cat_feat_undum['mcc'].astype(float).astype(int)

    '''
    CONCATENATE ALL FEATURES
    '''
    synth_df = pd.concat([pd.DataFrame(synth_data_scaled_cl, columns=client_info), \
                        pd.DataFrame(synth_cont_feat, columns=cont_features),\
                        synth_time[:n_samples].reset_index(drop=True), synth_df_cat_feat_undum], axis=1)
    
    return synth_df, synth_df_cat


'''
SYNTHETIC TIME GENERATION
'''

def optimize_xi(xiP, delta, k, n):
    return mean_squared_error(np.mean(np.random.poisson(xiP, size=(k, n)), axis=0), delta)

def optimize_xi_by_deltas_split(deltas, n_splits):
    xiP_array = []
    idx_array = []
    k = 200

    for delta in deltas:
        
        quantiles = [np.quantile(delta, n * 1/n_splits) for n in range(n_splits + 1)]

        idx_quantiles = []

        for i in range(n_splits):
            idx_quantiles.append(list(np.where((quantiles[i] <= delta) & (delta < quantiles[i+1]))[0]))

        idx_quantiles.append(list(np.where(quantiles[i+1] <= delta)[0]))

        delta_array = []
        xi0_array = []
        bnds_array = []

        for i in idx_quantiles:
            if i != []:
                delta_array.append(delta[i])
                xi0_array.append(np.median(delta[i]))
                if np.median(delta[i]) == 0:
                    bnds_array.append([(0.0, 0.0)])
                else:    
                    bnds_array.append([(np.min(delta[i]), 2 * np.median(delta[i]))])
            else:
                delta_array.append(np.zeros(3))
                xi0_array.append(1)
                bnds_array.append([(0, 1)])

        xiP_array_delta = []

        for i in range(len(delta_array)):
            x_opt = minimize(lambda x: optimize_xi(x, delta_array[i], k, len(delta_array[i])), x0=[xi0_array[i]],\
                            tol=1e-6, bounds=bnds_array[i], options={"maxiter": 3000}, method='L-BFGS-B').x
            xiP_array_delta.append(x_opt)

        xiP_array.append(xiP_array_delta)
        idx_array.append(idx_quantiles)

    return xiP_array, idx_array

def generate_synth_deltas_poisson_split(deltas, n_splits, opt_time, xi_array, q_array):    
    synth_deltas = []

    if opt_time:
        xiP_array, idx_array = optimize_xi_by_deltas_split(deltas, n_splits)
    else:
        xiP_array = xi_array
        idx_array = q_array

    deltas = deltas.values
    
    for i in range(len(deltas)):
        synth_deltas_primary = []
        for j in range(len(xiP_array[i])):
            if idx_array[i][j] != []:
                synth_deltas_primary.append(np.around(np.mean(np.random.poisson(xiP_array[i][j][0], (200, len(idx_array[i][j]))), axis=0)))
            else:
                continue

        synth = sorted(list(zip(np.hstack(idx_array[i]), np.hstack(synth_deltas_primary))), key=lambda x: x[0])

        synth_deltas.append(np.array(synth).T[1])

    synth_deltas_by_clients = synth_deltas

    return np.hstack(synth_deltas).astype(int), synth_deltas_by_clients, xiP_array, idx_array

def generate_synth_time(data, client_id, time_id, model='normal', n_splits=2, opt_time=True, xi_array=[], q_array=[]):
    
    deltas_by_clients = data.groupby(client_id)[time_id].apply(lambda x: (x - x.shift()).dt.days.values[1:])
    first_dates_by_clients = data.groupby(client_id)[time_id].first().values
    length_dates_by_clients = data.groupby(client_id)[time_id].count().values - 1

    deltas = np.hstack(deltas_by_clients)

    if model == 'poisson':
        synth_deltas, synth_deltas_by_clients, xiP_array, idx_array = generate_synth_deltas_poisson_split(deltas_by_clients,\
                                                                                    n_splits, opt_time, xi_array, q_array)

    elif model == 'normal':
        synth_deltas = abs(deltas + np.around(np.random.normal(0, 0.5, len(deltas))).astype(int))
        synth_deltas_by_clients = synth_deltas
        xiP_array = []
        idx_array = []
    
    else:
        print('Choose the model for synthetic time generation')

    splitted_synth_deltas = np.array(np.split(synth_deltas.astype('timedelta64[D]'), np.cumsum(length_dates_by_clients))[:-1])
    synth_dates_by_clients = list(map(list, first_dates_by_clients.reshape(-1, 1)))

    for i in range(len(splitted_synth_deltas)):
        for j in range(len(splitted_synth_deltas[i])):
            synth_dates_by_clients[i].append(splitted_synth_deltas[i][j] + synth_dates_by_clients[i][j])
    
    synth_time = pd.DataFrame(np.hstack(np.array(synth_dates_by_clients)), columns=[time_id]).sort_values(by=time_id).reset_index(drop=True)

    return synth_time, deltas_by_clients, synth_deltas_by_clients, xiP_array, idx_array

# def optimize_xi_by_deltas(deltas):
#     xiP_array = []
#     k = 100

#     for delta in deltas:
#         if not np.any(delta):
#             delta = np.array([1])

#         bnds = [(0, 2 * delta.mean())]
#         xiP_array.append(minimize(lambda x: optimize_xi(x, delta, k, len(delta)), x0=[delta.mean()], tol=1e-6, bounds=bnds, options={"maxiter": 5000}).x)

#     return xiP_array

# def generate_synth_deltas_poisson(deltas):    
#     synth_deltas = []
#     xiP_array = optimize_xi_by_deltas(deltas)
#     deltas = deltas.values
    
#     for i in range(len(deltas)):
#         # xiP = deltas[i].mean()
#         synth_deltas.append(np.random.poisson(xiP_array[i][0], len(deltas[i])))

#     return np.hstack(synth_deltas).astype(int)


# def generate_synth_time(data, client_id, time_id, model='normal'):
    
#     deltas_by_clients = data.groupby(client_id)[time_id].apply(lambda x: (x - x.shift()).dt.days.values[1:])
#     first_dates_by_clients = data.groupby(client_id)[time_id].first().values
#     length_dates_by_clients = data.groupby(client_id)[time_id].count().values - 1

#     deltas = np.hstack(deltas_by_clients)

#     if model == 'poisson':
#         synth_deltas = generate_synth_deltas_poisson(deltas_by_clients)

#     elif model == 'normal':
#         synth_deltas = deltas + np.around(np.random.normal(0, 0.7, len(deltas))).astype(int)
    
#     else:
#         print('Choose the model for synthetic time generation')

#     splitted_synth_deltas = np.array(np.split(synth_deltas.astype('timedelta64[D]'), np.cumsum(length_dates_by_clients))[:-1])
#     synth_dates_by_clients = list(map(list, first_dates_by_clients.reshape(-1, 1)))

#     for i in range(len(splitted_synth_deltas)):
#         for j in range(len(splitted_synth_deltas[i])):
#             synth_dates_by_clients[i].append(splitted_synth_deltas[i][j] + synth_dates_by_clients[i][j])
    
#     synth_time = pd.DataFrame(np.hstack(np.array(synth_dates_by_clients)), columns=[time_id])

#     return synth_time


'''
SCENARIO MODELLING
'''

def change_scenario(X_oh, mcc, data, rate):
    X_oh = copy.deepcopy(X_oh)

    idx_scenario_1 = np.where(X_oh[mcc] == 1)[0]
    idx_scenario_1_compl = np.setdiff1d(X_oh.index, idx_scenario_1)

    new_idx_sc_1 = random.sample(list(idx_scenario_1_compl), rate)

    X_oh.loc[new_idx_sc_1, mcc] = np.ones(len(new_idx_sc_1)).astype(int)
    X_oh = X_oh.astype(int)

    X_oh.loc[new_idx_sc_1, np.setdiff1d(X_oh.iloc[:, :len(data['mcc'].unique())].columns, mcc)] = 0

    return X_oh

# def create_cond_vector_scenario(X_oh_sc, encoder_onehot, date_transformations, behaviour_cl_enc, encoder, X_cont, X_cl, scaler):
#     X_oh_emb = encoder_onehot(torch.FloatTensor(X_oh_sc.values).to(device)).detach().numpy()

#     data_transformed = np.concatenate([X_cont, X_oh_emb], axis=1)
#     data_transformed = scaler.transform(data_transformed)
#     data_transformed = np.concatenate([X_cl, data_transformed], axis=1)

#     X_emb = encoder(torch.FloatTensor(data_transformed).to(device)).detach().cpu().numpy()
    
#     cond_vector = np.concatenate([X_emb, date_transformations, behaviour_cl_enc], axis=1)

#     return cond_vector


def change_scenario_rnf(X_oh, value, mcc_by_values, data, rate):
    X_oh = copy.deepcopy(X_oh)

    queue_idx = X_oh.index

    for mcc in mcc_by_values[value]:
        mcc = 'mcc_' + str(mcc)

        idx_scenario_1 = np.where(X_oh[mcc] == 1)[0]
        idx_scenario_1_compl = np.setdiff1d(queue_idx, idx_scenario_1)
        queue_idx = idx_scenario_1_compl

        new_idx_sc_1 = random.sample(list(idx_scenario_1_compl), rate)

        X_oh.loc[new_idx_sc_1, mcc] = np.ones(len(new_idx_sc_1)).astype(int)
        X_oh = X_oh.astype(int)

        X_oh.loc[new_idx_sc_1, np.setdiff1d(X_oh.iloc[:, :len(data['mcc'].unique())].columns, mcc)] = 0

    return X_oh

def sample_scenario(n_samples, generator, supervisor, noise_dim, cond_vector, X_oh_sc, scaler, X_cl, X_cont, encoder_onehot, encoder, data, behaviour_cl_enc,\
            date_feature, name_client_id, time='synth', model_time='poisson', n_splits=2, opt_time=True, xi_array=[], q_array=[], device='cpu'):
    
    X_oh_emb = encoder_onehot(torch.FloatTensor(X_oh_sc.values).to(device)).detach().numpy()

    data_transformed = np.concatenate([X_cont, X_oh_emb], axis=1)
    data_transformed = scaler.transform(data_transformed)
    data_transformed = np.concatenate([X_cl, data_transformed], axis=1)

    X_emb_cv = encoder(torch.FloatTensor(data_transformed).to(device)).detach().cpu().numpy()

    if n_samples <= len(cond_vector):
        synth_time, cond_vector_new = sample_cond_vector_with_time(n_samples, len(cond_vector), X_emb_cv, data,\
                    behaviour_cl_enc, date_feature, name_client_id, time, model_time, n_splits, opt_time, xi_array, q_array)
        
        noise = torch.FloatTensor(dclProcess(n_samples - 1, noise_dim)).to(device)
        z = torch.cat([noise.to(device), torch.FloatTensor(cond_vector_new[:n_samples]).to(device)], axis=1).to(device)
        synth_data = supervisor(torch.cat([generator(z).detach(), torch.FloatTensor(cond_vector_new[:n_samples]).to(device)], dim=1)).detach().cpu().numpy()

    else:
        synth_time, cond_vector_new = sample_cond_vector_with_time(n_samples, len(cond_vector), X_emb_cv, data,\
                    behaviour_cl_enc, date_feature, name_client_id, time, model_time, n_splits, opt_time, xi_array, q_array)
        
        noise = torch.FloatTensor(dclProcess(n_samples - 1, noise_dim)).to(device)
        z = torch.cat([noise.to(device), torch.FloatTensor(cond_vector_new).to(device)], axis=1).to(device)
        synth_data = supervisor(torch.cat([generator(z).detach(), torch.FloatTensor(cond_vector_new).to(device)], dim=1)).detach().cpu().numpy()

    return synth_data, synth_time




def dclProcess(N, M):
    T = 10
    theta = 15
    delta = 20

    Z1 = np.random.normal(0.0, 1.0, [M, N])
    X = np.zeros([M, N + 1])

    X[:, 0] = np.random.normal(0.0, 0.2, M)

    time = np.zeros([N+1])
    dt = T / float(N)
    
    for i in range(0, N):

        X[:,i+1] = X[:, i] - 1/theta * X[:,i] * dt + np.sqrt((1 - (X[:, i])**2)/(theta * (delta + 1))) * np.sqrt(dt) * Z1[:,i]
            
        if (X[:,i+1] > 1).any():
            X[np.where(X[:,i+1] >= 1)[0], i+1] = 0.9999

        if (X[:,i+1] < -1).any():
            X[np.where(X[:,i+1] <= -1)[0], i+1] = -0.9999 
            
        time[i+1] = time[i] + dt

    return X.T
