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
from rdt.transformers.numerical import GaussianNormalizer
from scipy import signal
from functorch import vmap

from TRGAN.encoders import *
from TRGAN.TRGAN_light_preprocessing import *


'''
ONEHOT FEATURES (small numbers of the unique categories)
'''

def create_onehot(data: pd.DataFrame, cat_features):
    data_cat_onehot = pd.get_dummies(data[cat_features], columns=cat_features)
    return data_cat_onehot


def encode_onehot_embeddings(data: pd.DataFrame, latent_dim, lr=1e-3, epochs=100, batch_size=2**8, device='cpu'):
    dim_onehot = len(data.columns)

    encoder_onehot = Encoder_onehot(dim_onehot, latent_dim).to(device)
    decoder_onehot = Decoder_onehot(latent_dim, dim_onehot).to(device)

    optimizer_Enc = optim.Adam(encoder_onehot.parameters(), lr=lr)
    optimizer_Dec = optim.Adam(decoder_onehot.parameters(), lr=lr)

    scheduler_Enc = torch.optim.lr_scheduler.ExponentialLR(optimizer_Enc, gamma=0.98)
    scheduler_Dec = torch.optim.lr_scheduler.ExponentialLR(optimizer_Dec, gamma=0.98)

    loader_onehot = DataLoader(torch.FloatTensor(data.values), batch_size, shuffle=True)

    epochs = tqdm(range(epochs))

    loss = torch.nn.BCELoss()

    for _ in epochs:
        for _, X in enumerate(loader_onehot):

            H = encoder_onehot(X.float().to(device))
            X_tilde = decoder_onehot(H.to(device))

            criterion = (loss(X_tilde, X.to(device))).to(device)
            
            optimizer_Enc.zero_grad()
            optimizer_Dec.zero_grad()
            
            criterion.backward()
            
            optimizer_Enc.step()
            optimizer_Dec.step()

        scheduler_Enc.step()
        scheduler_Dec.step()

        epochs.set_description(f'Loss E_oh: {criterion.item()}')

    encoder_onehot.eval()
    decoder_onehot.eval()

    data_cat_encode = encoder_onehot(torch.FloatTensor(data.values).to(device)).detach().cpu().numpy()
    
    scaler_onehot = {'encoder': encoder_onehot, 'decoder': decoder_onehot}

    return data_cat_encode, scaler_onehot


def decode_onehot_embeddings(onehot_embeddings: np.array, onehot_cols, scaler_onehot, mcc_name: str) -> pd.DataFrame:
    onehot_decoded = np.abs(np.around(scaler_onehot['decoder'](torch.FloatTensor(onehot_embeddings).to('cpu')).detach().cpu().numpy())).astype(int)
    df_onehot = pd.DataFrame(onehot_decoded, columns = onehot_cols)
    df_onehot = undummify(df_onehot, prefix_sep="_")
    df_onehot[mcc_name] = df_onehot[mcc_name].astype(int).astype(str)
    
    return df_onehot

'''
CONTINUOUS FEATURES
'''
def encode_continuous_embeddings(X, feat_names, type_scale='Autoencoder', epochs=100, lr=1e-3, bs=2**8, latent_dim=5, device='cpu'):
    data = copy.deepcopy(X)
    processing_dict = dict()

    if type_scale == 'Standardize':

        scaler_std = StandardScaler()
        data[feat_names] = scaler_std.fit_transform(data[feat_names].values)
        X_cont = data[feat_names].values
        processing_dict['scaler'] = scaler_std

    elif type_scale == 'Autoencoder':       
        gaus_tr = []
        
        # for col in feat_names:
        #     scaler_cont = GaussianNormalizer(learn_rounding_scheme=True, enforce_min_max_values=True)
        #     scaler_cont.reset_randomization()
        #     data[col] = scaler_cont.fit_transform(data, column=[col])[col]
        #     gaus_tr.append(scaler_cont)
            
        scaler_std = StandardScaler()
        data[feat_names] = scaler_std.fit_transform(data[feat_names].values)
        gaus_tr.append(scaler_std) 
            
            
        scaler_cont2 = MinMaxScaler((-1, 1))
        data[feat_names] = scaler_cont2.fit_transform(data[feat_names])

        encoder_cont_emb = Encoder_cont_emb(len(feat_names), latent_dim).to(device)
        decoder_cont_emb = Decoder_cont_emb(latent_dim, len(feat_names)).to(device)

        optimizer_Enc_cont_emb = optim.Adam(encoder_cont_emb.parameters(), lr, betas=(0.9, 0.999), amsgrad=True)
        optimizer_Dec_cont_emb = optim.Adam(decoder_cont_emb.parameters(), lr, betas=(0.9, 0.999), amsgrad=True)

        scheduler_Enc = torch.optim.lr_scheduler.ExponentialLR(optimizer_Enc_cont_emb, gamma=0.98)
        scheduler_Dec = torch.optim.lr_scheduler.ExponentialLR(optimizer_Dec_cont_emb, gamma=0.98)

        loader_cont_emb = DataLoader(torch.FloatTensor(data[feat_names].values), bs, shuffle=True)

        epochs = tqdm(range(epochs))
        # loss = torch.nn.HuberLoss()
        loss = torch.nn.MSELoss()

        for _ in epochs:
            for _, X in enumerate(loader_cont_emb):

                H = encoder_cont_emb(X.float().to(device))
                X_tilde = decoder_cont_emb(H.to(device))
                
                loss_mse = loss(X.float().to(device), X_tilde).to(device)
                
                optimizer_Enc_cont_emb.zero_grad()
                optimizer_Dec_cont_emb.zero_grad()
                
                loss_mse.backward()
                
                optimizer_Enc_cont_emb.step()
                optimizer_Dec_cont_emb.step()

            scheduler_Enc.step()
            scheduler_Dec.step()

            epochs.set_description(f'Loss E_cont: {loss_mse.item()}')

        encoder_cont_emb.eval()
        decoder_cont_emb.eval()

        X_cont = encoder_cont_emb(torch.FloatTensor(data[feat_names].values).to(device)).detach().cpu().numpy()
        
        processing_dict['decoder'] = decoder_cont_emb
        processing_dict['scaler_minmax'] = scaler_cont2
        processing_dict['scaler'] = gaus_tr
        processing_dict['encoder'] = encoder_cont_emb
        # scaler.append(decoder_cont_emb)
        # scaler.append(scaler_cont2)
        # scaler.append(gaus_tr)
        # scaler.append(encoder_cont_emb)

    else:
        print('Choose preprocessing scheme for continuous features. Available: CBNormalize and Standardize')


    # scaler = np.array(scaler, dtype=object)
    
    return X_cont, processing_dict



def decode_continuous_embeddings(embeddings: np.array, feat_names: list, scaler: dict, type_scale_cont: str = 'Autoencoder', device='cpu') -> pd.DataFrame:
    if type_scale_cont == 'Standardize':
        synth_cont_feat = embeddings
        synth_cont_feat = scaler['scaler'].inverse_transform(synth_cont_feat)

    elif type_scale_cont == 'Autoencoder':
        synth_cont_feat = embeddings
        synth_cont_feat = (scaler['decoder'](torch.FloatTensor(synth_cont_feat).to(device))).detach().cpu().numpy()
        synth_cont_feat = scaler['scaler_minmax'].inverse_transform(synth_cont_feat)
        
        # decoded_array = []
        # for i in range(len(feat_names)):
        #     scaler[2][i].reset_randomization()
        #     temp = scaler[2][i].reverse_transform(pd.DataFrame(synth_cont_feat[:, i], columns=[feat_names[i]]))
        #     decoded_array.append(temp.values)
        # synth_cont_feat = decoded_array
        
        synth_cont_feat = scaler['scaler'][0].inverse_transform(synth_cont_feat)

    else:
        print('Incorrect preprocessing type for continuous features')
    
    return pd.DataFrame(synth_cont_feat, columns=feat_names)

# '''
# DATE FEATURES
# '''

# def preprocessing_date(data: pd.DataFrame, date_feature: str) -> np.array:
#     min_year = np.min(data[date_feature].apply(lambda x: x.year))
#     max_year = np.max(data[date_feature].apply(lambda x: x.year))

#     date_transformations = data[date_feature].apply(lambda x: np.array([np.cos(2*np.pi * x.day / 30),\
#                                                                  np.sin(2*np.pi * x.day / 30),\
#                                           np.cos(2*np.pi * x.month / 12), np.sin(2*np.pi * x.month / 12),\
#                                           (x.year - min_year)/(max_year - min_year + 1e-7)])).values
    
#     date_transformations = np.vstack(date_transformations)
#     # date_transformations = date_transformations[:,:-1] #временно пока не придумаем что делать с годом

#     return date_transformations

'''
CATEGORICAL FEATURES
'''

def encode_categorical_embeddings(data: pd.DataFrame, cat_feat_names, latent_dim=4, enc_type:str = 'Autoencoder',
                                  lr=1e-3, epochs=100, batch_size=2**8, device='cpu'):
    scaler_cat = {}
    
    if enc_type == 'Frequency':
        embeddings, scaler_cl, freq_enc = create_categorical_embeddings(data, cat_feat_names)
        scaler_cat['encoder'] = ''
        scaler_cat['decoder'] = ''
        scaler_cat['scaler'] = scaler_cl
        scaler_cat['freq_encoder'] = freq_enc
        
    elif enc_type == 'Autoencoder':
        categorical_emb, scaler_cl, freq_enc = create_categorical_embeddings(data, cat_feat_names)


        encoder = Encoder_client_emb(categorical_emb.shape[1], latent_dim).to(device)
        decoder = Decoder_client_emb(latent_dim, categorical_emb.shape[1]).to(device)

        optimizer_Enc = optim.Adam(encoder.parameters(), lr)
        optimizer_Dec = optim.Adam(decoder.parameters(), lr)
        
        scheduler_Enc = torch.optim.lr_scheduler.ExponentialLR(optimizer_Enc, gamma=0.98)
        scheduler_Dec = torch.optim.lr_scheduler.ExponentialLR(optimizer_Dec, gamma=0.98)

        loader_cl_emb = DataLoader(torch.FloatTensor(categorical_emb), batch_size, shuffle=True)

        epochs = tqdm(range(epochs))
        loss = torch.nn.MSELoss()

        for _ in epochs:
            for _, X in enumerate(loader_cl_emb):

                H = encoder(X.float().to(device))
                X_tilde = decoder(H.to(device))
                
                loss_mse = loss(X.float().to(device), X_tilde).to(device)
                
                optimizer_Enc.zero_grad()
                optimizer_Dec.zero_grad()
                
                loss_mse.backward()
                
                optimizer_Enc.step()
                optimizer_Dec.step()
            
            scheduler_Enc.step()
            scheduler_Dec.step()

            epochs.set_description(f'Loss E_cl: {loss_mse.item()}')

        embeddings = encoder(torch.FloatTensor(categorical_emb).to(device)).detach().cpu().numpy()
        
        scaler_cat['encoder'] = encoder
        scaler_cat['decoder'] = decoder
        scaler_cat['scaler'] = scaler_cl
        scaler_cat['freq_encoder'] = freq_enc
        
    else:
        print('Choose encoding type')

    return embeddings, scaler_cat


def decode_categorical_embeddings(embeddings: np.array, cat_feat_names: list, scaler_cat: dict, enc_type:str = 'Autoencoder', device='cpu'):
    
    if enc_type == 'Frequency':
        dec_array = inverse_categorical_embeddings(embeddings, cat_feat_names, scaler_cat['scaler'], scaler_cat['freq_encoder'])
        df_cat =  pd.DataFrame(dec_array, columns=cat_feat_names)
        
    elif enc_type == 'Autoencoder':
        synth_data_scaled_cl = scaler_cat['decoder'](torch.FloatTensor(embeddings).to(device)).detach().cpu().numpy()
        
        dec_array = inverse_categorical_embeddings(synth_data_scaled_cl, cat_feat_names, scaler_cat['scaler'], scaler_cat['freq_encoder'])
        df_cat =  pd.DataFrame(dec_array, columns=cat_feat_names)
        
    return df_cat
    

'''
CREATE EMBEDDINGS AND CONDITIONAL VECTOR
'''
def create_embeddings(onehot_emb: np.array, categorical_emb: np.array, numerical_emb: np.array) -> np.array:
    embedding = np.concatenate([onehot_emb, categorical_emb, numerical_emb], axis=1)

    return embedding


def create_cond_vector(data: pd.DataFrame, X_emb: np.array, date_feature: str, name_client_id: str, time_type: str='synth', latent_dim: int=4,\
                    lr:float=1e-3, epochs:int=20, batch_size:int=2**8, model_time:str='poisson',\
                    n_splits:int=2, opt_time:bool=True, xi_array:list=[], q_array:list=[], device:str='cpu'):

    if time_type == 'synth':
        synth_time, deltas_by_clients, synth_deltas_by_clients, xiP_array, idx_array = generate_synth_time(data,\
                                        name_client_id, date_feature, model_time, n_splits, opt_time, xi_array, q_array)
        
        date_transform = preprocessing_date(synth_time, date_feature)

    elif time_type == 'initial':
        synth_time = data[date_feature]
        date_transform = preprocessing_date(data, date_feature)
        deltas_by_clients = 'Only when time_type="synth"'
        synth_deltas_by_clients = 'Only when time_type="synth"'
        xiP_array = []
        idx_array = []

    else:
        print('Choose time type generation type')

    data_dim = len(X_emb[0])

    encoder = Encoder(data_dim, latent_dim).to(device)
    decoder = Decoder(latent_dim, data_dim).to(device)

    optimizer_Enc = optim.Adam(encoder.parameters(), lr)
    optimizer_Dec = optim.Adam(decoder.parameters(), lr)

    loader = DataLoader(torch.FloatTensor(X_emb), batch_size=batch_size, shuffle=True)

    epochs = tqdm(range(epochs))
    loss = torch.nn.MSELoss()

    for _ in epochs:
        for _, X in enumerate(loader):

            H = encoder(X.float().to(device))
            X_tilde = decoder(H.to(device))
            
            loss_mse = loss(X.float().to(device), X_tilde).to(device)
            
            optimizer_Enc.zero_grad()
            optimizer_Dec.zero_grad()
            
            loss_mse.backward()
        
            optimizer_Enc.step()
            optimizer_Dec.step()

        epochs.set_description('Loss E_cv: %.9f'  % (loss_mse.item()))

    data_encode = encoder(torch.FloatTensor(X_emb).to(device)).detach().cpu().numpy()
    
    cond_vector = np.concatenate([data_encode, date_transform], axis=1)
    
    cv_params = {'date_transform': date_transform, 'encoder': encoder, 'deltas_real': deltas_by_clients,
                 'deltas_synth': synth_deltas_by_clients, 'xiP': xiP_array, 'quantile_index': idx_array}

    return cond_vector, synth_time, cv_params



def inverse_transform(synth_data:np.array, latent_dim_onehot:int, latent_dim_cat:int, latent_dim_num:int) -> pd.DataFrame:
    synth_df_onehot = decode_onehot_embeddings(synth_data[:, :latent_dim_onehot], X_oh.columns, decoder_onehot, mcc_name)
    synth_df_cat = decode_categorical_embeddings(synth_data[:, latent_dim_onehot:latent_dim_onehot+latent_dim_cat], cat_feat_names, decoder_cat, scaler_cat, freq_enc)
    synth_df_num = decode_continuous_embeddings(synth_data[:, -latent_dim_num:], num_feat_names, scaler_num)
    
    synth_df = pd.concat([synth_df_onehot, synth_df_cat, synth_df_num], axis=1)
    
    return synth_df


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

    # scheduler_G = torch.optim.lr_scheduler.ExponentialLR(optimizer_G, gamma=0.97)
    # scheduler_D = torch.optim.lr_scheduler.ExponentialLR(optimizer_D, gamma=0.97)
    # scheduler_S = torch.optim.lr_scheduler.ExponentialLR(optimizer_S, gamma=0.97)
    # scheduler_D2 = torch.optim.lr_scheduler.ExponentialLR(optimizer_D2, gamma=0.97)

    data_with_cv = torch.cat([torch.FloatTensor(X_emb), torch.FloatTensor(cond_vector)], axis=1)

    idx_batch_array = np.arange(len(data_with_cv)//batch_size * batch_size)
    last_idx = np.setdiff1d(np.arange(len(data_with_cv)), idx_batch_array)
    split_idx = np.split(idx_batch_array, batch_size)
    split_idx_perm = np.random.permutation(split_idx)
    split_idx_perm = np.append(split_idx_perm, last_idx)

    loader_g = DataLoader(data_with_cv[split_idx_perm], batch_size=batch_size, shuffle=True)

    epochs = tqdm(range(num_epochs))
    loss_array = []

    b_d1 = 0.02
    b_d2 = 0.02

    for _ in epochs:
        for batch_idx, X in enumerate(loader_g):
            loss = torch.nn.MSELoss()
            batch_size = X.size(0)

            Vc = X[:, -dim_Vc:].to(device)
            
            # noise = torch.randn(batch_size, dim_noise).to(device)
            noise = torch.FloatTensor(dclProcess(batch_size - 1, dim_noise)).to(device)
            z = torch.cat([noise, Vc], dim=1).to(device)
            
            fake = torch.nan_to_num(generator(z))
            X = X.to(device)
            
            discriminator.trainable = True
            
            disc_loss = (-torch.mean(torch.nan_to_num(discriminator(X))) + torch.mean(torch.nan_to_num(discriminator(torch.cat([fake, Vc], dim=1))))).to(device) +\
                grad_penalty(discriminator, X, torch.cat([fake, Vc], dim=1), device)
     
            fake_super = supervisor(torch.cat([fake.detach(), Vc], dim=1)).to(device)
            disc2_loss = (-torch.mean(discriminator2(X)) + torch.mean(discriminator2(torch.cat([fake_super, Vc], dim=1)))).to(device) +\
                grad_penalty(discriminator2, X, torch.cat([fake_super, Vc], dim=1), device)

            optimizer_D.zero_grad()
            disc_loss.backward()
            optimizer_D.step()

            optimizer_D2.zero_grad()
            disc2_loss.backward()
            optimizer_D2.step()

            # for dp in discriminator.parameters():
            #             dp.data.clamp_(-b_d1, b_d1)

            # for dp in discriminator2.parameters():
            #             dp.data.clamp_(-b_d2, b_d2)

                    
            if batch_idx % 3 == 0:
                discriminator.trainable = False

                gen_loss1 = -torch.mean(torch.nan_to_num(discriminator(torch.cat([generator(z), Vc], dim=1)))).to(device)
                supervisor_loss = (-torch.mean(discriminator2(torch.cat([supervisor(torch.cat([generator(z), Vc], dim=1)), Vc], dim=1))) +\
                                    lambda1 * loss(supervisor(torch.cat([generator(z), Vc], dim=1).detach()), X[:,:-dim_Vc])).to(device)
                
                # gen_loss1 = torch.mean(torch.log(discriminator(torch.cat([generator(z), Vc], dim=1)))).to(device)
                # supervisor_loss = (torch.mean(torch.log(discriminator2(torch.cat([supervisor(torch.cat([generator(z), Vc], dim=1).detach()), Vc], dim=1)))) +\
                #                     lambda1 * loss(supervisor(torch.cat([generator(z), Vc], dim=1).detach()), X[:,:-dim_Vc])).to(device)
                
                gen_loss = (alpha * gen_loss1 + (1 - alpha) * supervisor_loss)
                
                
                supervisor_loss2 = ((-torch.mean(discriminator2(torch.cat([supervisor(torch.cat([generator(z).detach(), Vc], dim=1)),\
                    Vc], dim=1))))  + lambda1 * loss(supervisor(torch.cat([generator(z), Vc], dim=1).detach()), X[:,:-dim_Vc])).to(device)
                
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
        
        decoded_array = []
        for i in range(len(cont_features)):
            scaler_cont[2][i].reset_randomization()
            temp = scaler_cont[2][i].reverse_transform(pd.DataFrame(synth_cont_feat[:, i], columns=[cont_features[i]]))
            decoded_array.append(temp.values)
        
        synth_cont_feat = decoded_array
        # synth_cont_feat = scaler_cont[2].reverse_transform(pd.DataFrame(synth_cont_feat, columns=cont_features))

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


    splitted_synth_deltas = np.split(synth_deltas.astype('timedelta64[D]'), np.cumsum(length_dates_by_clients))[:-1]
    synth_dates_by_clients = list(map(list, first_dates_by_clients.reshape(-1, 1)))

    for i in range(len(splitted_synth_deltas)):
        for j in range(len(splitted_synth_deltas[i])):
            synth_dates_by_clients[i].append(splitted_synth_deltas[i][j] + synth_dates_by_clients[i][j])
    
    synth_time = pd.DataFrame(np.hstack(np.array(synth_dates_by_clients, dtype='object')), columns=[time_id]).sort_values(by=time_id).reset_index(drop=True)

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
