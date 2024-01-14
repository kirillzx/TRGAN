import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy

import torch
from torch import optim
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.special import rel_entr, kl_div
from scipy.spatial.distance import jensenshannon
from scipy.stats import kstest, ks_2samp
import scipy.stats as sts
from tqdm import tqdm
from sdv.metadata import SingleTableMetadata

from TRGAN.TRGAN_main import *
import TRGAN.Privacy_modules as privacy_trgan
from TRGAN.encoders import *
import TRGAN.TRGAN_train_load_modules as trgan_train
from TRGAN.evaluation_metrics import *
from TRGAN.methods_comparison import *
from TRGAN.dpgan import *

import warnings
warnings.filterwarnings("ignore")

plt.style.use('ggplot')
import matplotlib as mpl
#set params for the article
# mpl.rcParams['xtick.labelsize'] = 16
# mpl.rcParams['ytick.labelsize'] = 16
# mpl.rcParams['legend.fontsize'] = 14
# mpl.rcParams['axes.labelsize'] = 18

#set params for the notebook
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['legend.fontsize'] = 10
mpl.rcParams['axes.labelsize'] = 14


def train_all_models(load=False, experiment_id='Privacy_RAIF_2_eps=0.5'):

    '''
    IMPORT DATA
    '''

    data = pd.read_csv('Data/data_raif_clean.csv')
    data = data[['transaction_date', 'customer', 'mcc', 'amount']]
    data['transaction_date'] = pd.to_datetime(data['transaction_date'], format='%Y-%m-%d')

    '''
    FEATURES NAMES
    '''
    cat_features = ['mcc']
    cont_features = ['amount']
    date_feature = ['transaction_date']
    client_info = ['customer']

    # data_cat = data[cat_features] 

    '''
    DIMENSIONS
    '''
    dim_X_cat = len(cat_features)
    dim_cont_emb = 1
    dim_X_cont = dim_cont_emb * len(cont_features)
    dim_X_date = 4
    dim_Xoh = 20 # dimension of X one hot embeddings
    dim_Xcl = 4  # dimension of client's info embeddings
    dim_Vc_h = 10 # dimension of conditional vector
    dim_bce = 5 # dimension of the behaviour client encoding
    dim_Vc = dim_Vc_h + dim_X_date
    dim_X_emb = dim_Xoh + dim_X_cont + dim_Xcl
    dim_noise = 25
    # data_dim = len(data_transformed[0])

    '''
    LEARNING RATES
    '''
    lr_E_oh = 3e-4
    lr_E_cl = 3e-4
    lr_E_Vc = 3e-4
    lr_E_cont = 3e-4

    '''
    SAVE DIRECTORY
    '''
    DIRECTORY = 'Pretrained_model/'
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    EPSILON = 0.01
    EPSILON_oh = 5


    X_oh = onehot_emb_categorical(data, cat_features)
    X_oh_emb, encoder_onehot, decoder_onehot = trgan_train.create_cat_emb(X_oh, dim_Xoh, lr_E_oh, epochs=20,\
                batch_size=2**7, load=load, directory=DIRECTORY, names=[f'TRGAN_E_oh_exp_{experiment_id}.pkl',\
                f'TRGAN_D_oh_exp_{experiment_id}.pkl', f'X_oh_emb_exp_{experiment_id}.npy'], device=DEVICE, eps=EPSILON_oh)

    X_cont, scaler_cont = trgan_train.create_cont_emb(dim_X_cont, data, cont_features, lr_E_cont, epochs=20, batch_size=2**7,\
                load=load, directory='Pretrained_model/', names=f'scaler_cont_{experiment_id}.npy', type_scale='CBNormalize', device=DEVICE, eps=EPSILON)


    X_cl, encoder_cl_emb, decoder_cl_emb, scaler_cl_emb, label_encoders = trgan_train.create_client_emb(dim_Xcl, data, client_info,\
                dim_Xcl, lr_E_cl, epochs=20, batch_size=2**7, load=load, directory=DIRECTORY, names=[f'TRGAN_E_cl_{experiment_id}.pkl',\
                f'TRGAN_D_cl_{experiment_id}.pkl', f'X_cl_{experiment_id}.npy', f'scaler_{experiment_id}.joblib', f'label_enc_{experiment_id}.joblib'],\
                device=DEVICE, eps=EPSILON)


    X_emb, scaler_emb = create_embeddings(X_cont, X_oh_emb, X_cl)

    cond_vector, synth_time, date_transformations, behaviour_cl_enc, encoder, deltas_by_clients, synth_deltas_by_clients, xiP_array, idx_array =\
                trgan_train.create_conditional_vector(data, X_emb, date_feature, 'initial', dim_Vc_h, dim_bce, \
                name_client_id='customer', name_agg_feature='amount', lr_E_Vc=lr_E_Vc, epochs=15, batch_size=2**7, model_time='poisson', n_splits=4, load=load,\
                directory=DIRECTORY, names=[f'TRGAN_E_Vc_{experiment_id}.pkl', f'Vc_{experiment_id}.npy', f'BCE_{experiment_id}.npy'], opt_time=True,\
                device=DEVICE, eps=EPSILON)
                


    number_of_experiments = 2
    synth_data_trgan_arr = []
    synth_data_ae_arr = []
    synth_data_rdp_arr = []

    for i in range(number_of_experiments):
        '''
        TRGAN
        '''
        h_dim = 2**6
        num_blocks_gen = 1
        num_blocks_dis = 1
        gauss_filter_dim = 20

        if load:
            generator = privacy_trgan.Generator(dim_noise + dim_Vc, dim_X_emb, h_dim, num_blocks_gen, gauss_filter_dim, DEVICE).to(DEVICE)
            supervisor = privacy_trgan.Supervisor(dim_X_emb + dim_Vc, dim_X_emb, h_dim, num_blocks_gen, gauss_filter_dim, DEVICE).to(DEVICE)

            generator.load_state_dict(torch.load(f'{DIRECTORY}TRGAN_generator_exp_{experiment_id}.pkl'))
            supervisor.load_state_dict(torch.load(f'{DIRECTORY}TRGAN_supervisor_exp_{experiment_id}.pkl'))

            generator.eval()
            supervisor.eval()

            loss_array = np.load(f'{DIRECTORY}loss_array_exp_{experiment_id}.npy')

        else:
            generator, supervisor, loss_array, discriminator, discriminator2, epsilon_bar_array = privacy_trgan.train_generator(X_emb, cond_vector,\
                                    dim_Vc, dim_X_emb, dim_noise, batch_size=2**8, lr_rates=[3e-4, 3e-4, 3e-4, 3e-4], num_epochs=30, num_blocks_gen=num_blocks_gen,\
                                    num_blocks_dis=num_blocks_dis, h_dim=h_dim, lambda1=3, alpha_r=0.75, window_size=gauss_filter_dim, device=DEVICE,\
                                    privacy_type='TDDP', eps=0.5)

            torch.save(generator.state_dict(), f'{DIRECTORY}TRGAN_generator_exp_{experiment_id}.pkl')
            torch.save(supervisor.state_dict(), f'{DIRECTORY}TRGAN_supervisor_exp_{experiment_id}.pkl')

            np.save(f'{DIRECTORY}loss_array_exp_{experiment_id}.npy', loss_array)

            generator.eval()
            supervisor.eval()


        n_samples = len(X_emb)
        synth_data, synth_time = sample(n_samples, generator, supervisor, dim_noise, cond_vector, X_emb, encoder, data, behaviour_cl_enc,\
                                    date_feature, 'customer', time='initial', model_time='poisson', n_splits=4, opt_time=False,\
                                    xi_array=xiP_array, q_array=idx_array, device=DEVICE)

        

        '''
        TRGAN with differential private autoencoders
        '''
        h_dim = 2**6
        num_blocks_gen = 1
        num_blocks_dis = 1
        gauss_filter_dim = 20

        if load:
            generator_ae = Generator(dim_noise + dim_Vc, dim_X_emb, h_dim, num_blocks_gen, gauss_filter_dim, DEVICE).to(DEVICE)
            supervisor_ae = Supervisor(dim_X_emb + dim_Vc, dim_X_emb, h_dim, num_blocks_gen, gauss_filter_dim, DEVICE).to(DEVICE)

            generator_ae.load_state_dict(torch.load(f'{DIRECTORY}TRGAN_generator_ae_exp_{experiment_id}.pkl'))
            supervisor_ae.load_state_dict(torch.load(f'{DIRECTORY}TRGAN_supervisor_ae_exp_{experiment_id}.pkl'))

            generator_ae.eval()
            supervisor_ae.eval()

            loss_array_ae = np.load(f'{DIRECTORY}loss_array_ae_{experiment_id}.npy')

        else:
            generator_ae, supervisor_ae, loss_array_ae, _, _ = train_generator(X_emb, cond_vector,\
                                    dim_Vc, dim_X_emb, dim_noise, batch_size=2**8, lr_rates=[3e-4, 3e-4, 3e-4, 3e-4], num_epochs=30, num_blocks_gen=num_blocks_gen,\
                                    num_blocks_dis=num_blocks_dis, h_dim=h_dim, lambda1=3, alpha=0.8, window_size=gauss_filter_dim, device=DEVICE)

            torch.save(generator_ae.state_dict(), f'{DIRECTORY}TRGAN_generator_ae_exp_{experiment_id}.pkl')
            torch.save(supervisor_ae.state_dict(), f'{DIRECTORY}TRGAN_supervisor_ae_exp_{experiment_id}.pkl')

            np.save(f'{DIRECTORY}loss_array_ae_{experiment_id}.npy', loss_array_ae)

            generator_ae.eval()
            supervisor_ae.eval()

        n_samples = len(X_emb)
        synth_data_ae, synth_time_ae = sample(n_samples, generator_ae, supervisor_ae, dim_noise, cond_vector, X_emb, encoder, data, behaviour_cl_enc,\
                                    date_feature, 'customer', time='initial', model_time='poisson', n_splits=4, opt_time=False,\
                                    xi_array=xiP_array, q_array=idx_array, device=DEVICE)

        


        h_dim = 2**6
        num_blocks_gen = 1
        num_blocks_dis = 1
        gauss_filter_dim = 20

        if load:
            generator_rdp = privacy_trgan.Generator(dim_noise + dim_Vc, dim_X_emb, h_dim, num_blocks_gen, gauss_filter_dim, DEVICE).to(DEVICE)
            supervisor_rdp = privacy_trgan.Supervisor(dim_X_emb + dim_Vc, dim_X_emb, h_dim, num_blocks_gen, gauss_filter_dim, DEVICE).to(DEVICE)

            generator_rdp.load_state_dict(torch.load(f'{DIRECTORY}TRGAN_generator_rdp_exp_{experiment_id}.pkl'))
            supervisor_rdp.load_state_dict(torch.load(f'{DIRECTORY}TRGAN_supervisor_rdp_exp_{experiment_id}.pkl'))

            generator_rdp.eval()
            supervisor_rdp.eval()

            loss_array_rdp = np.load(f'{DIRECTORY}loss_array_rdp_exp_{experiment_id}.npy')

        else:
            generator_rdp, supervisor_rdp, loss_array_rdp, _, _, _ = privacy_trgan.train_generator(X_emb, cond_vector,\
                                    dim_Vc, dim_X_emb, dim_noise, batch_size=2**8, lr_rates=[3e-4, 3e-4, 3e-4, 3e-4], num_epochs=30, num_blocks_gen=num_blocks_gen,\
                                    num_blocks_dis=num_blocks_dis, h_dim=h_dim, lambda1=3, alpha_r=0.7, window_size=gauss_filter_dim, device=DEVICE,\
                                    privacy_type='RDP', eps=0.5)

            torch.save(generator_rdp.state_dict(), f'{DIRECTORY}TRGAN_generator_rdp_exp_{experiment_id}.pkl')
            torch.save(supervisor_rdp.state_dict(), f'{DIRECTORY}TRGAN_supervisor_rdp_exp_{experiment_id}.pkl')

            np.save(f'{DIRECTORY}loss_array_rdp_exp_{experiment_id}.npy', loss_array_rdp)

            generator_rdp.eval()
            supervisor_rdp.eval()


        n_samples = len(X_emb)
        synth_data_rdp, synth_time_rdp = sample(n_samples, generator_rdp, supervisor_rdp, dim_noise, cond_vector, X_emb, encoder, data, behaviour_cl_enc,\
                                    date_feature, 'customer', time='initial', model_time='poisson', n_splits=4, opt_time=False,\
                                    xi_array=xiP_array, q_array=idx_array, device=DEVICE)
        
        synth_data_rdp_arr.append(synth_data_rdp)
        synth_data_trgan_arr.append(synth_data)
        synth_data_ae_arr.append(synth_data_ae)
        
        
    synth_data_rdp_arr = np.array(synth_data_rdp_arr)
    synth_data_trgan_arr = np.array(synth_data_trgan_arr)
    synth_data_ae_arr = np.array(synth_data_ae_arr)
        
    synth_data_rdp_arr = synth_data_rdp_arr[np.where(list(map(lambda x: not np.isnan(x).any(), synth_data_rdp_arr)))[0]]
    synth_data_trgan_arr = synth_data_trgan_arr[np.where(list(map(lambda x: not np.isnan(x).any(), synth_data_trgan_arr)))[0]]
    synth_data_ae_arr = synth_data_ae_arr[np.where(list(map(lambda x: not np.isnan(x).any(), synth_data_ae_arr)))[0]]

    synth_data_rdp = np.mean(synth_data_rdp_arr, axis=0)
    synth_data = np.mean(synth_data_trgan_arr, axis=0)
    synth_data_ae = np.mean(synth_data_ae_arr, axis=0)
        
        
    synth_df, synth_df_cat = inverse_transforms(n_samples, synth_data, synth_time, client_info, cont_features, X_oh, scaler_emb, scaler_cl_emb, scaler_cont,\
                        label_encoders, decoder_cl_emb, decoder_onehot, dim_Xcl, dim_X_cont, type_scale_cont='CBNormalize', device=DEVICE)


    synth_df_ae, _ = inverse_transforms(n_samples, synth_data_ae, synth_time_ae, client_info, cont_features, X_oh, scaler_emb, scaler_cl_emb, scaler_cont,\
                        label_encoders, decoder_cl_emb, decoder_onehot, dim_Xcl, dim_X_cont, type_scale_cont='CBNormalize', device=DEVICE)

    synth_df_rdp, synth_df_cat = inverse_transforms(n_samples, synth_data_rdp, synth_time_rdp, client_info, cont_features, X_oh, scaler_emb, scaler_cl_emb, scaler_cont,\
                        label_encoders, decoder_cl_emb, decoder_onehot, dim_Xcl, dim_X_cont, type_scale_cont='CBNormalize', device=DEVICE)



    '''
    DPGAN
    '''

    eps = 0.5

    loss_array_dpgan, synth_df_dpgan = sample_dpgan(X_emb, cond_vector, dim_Vc, dim_X_emb, dim_noise, encoder, data, behaviour_cl_enc, date_feature, xiP_array,\
                idx_array, client_info, cont_features, X_oh, scaler_emb, scaler_cl_emb, scaler_cont,\
                label_encoders, decoder_cl_emb, decoder_onehot, dim_Xcl, dim_X_cont, 'CBNormalize',
                'DPGAN', eps, load, DEVICE, DIRECTORY, experiment_id)


    '''
    TRGAN with DPGAN
    '''

    eps = 0.5

    loss_array_trgan_dpgan, synth_df_trgan_dpgan = sample_trgan_dpgan(X_emb, cond_vector, dim_Vc, dim_X_emb, dim_noise, encoder, data, behaviour_cl_enc, date_feature, xiP_array,\
                idx_array, client_info, cont_features, X_oh, scaler_emb, scaler_cl_emb, scaler_cont,\
                label_encoders, decoder_cl_emb, decoder_onehot, dim_Xcl, dim_X_cont, 'CBNormalize',
                'TRGAN_DPGAN', eps, load, DEVICE, DIRECTORY, experiment_id)


    '''
    PATE-GAN
    '''
    eps = 0.5

    loss_array_pategan, synth_df_pategan = sample_dpgan(X_emb, cond_vector, dim_Vc, dim_X_emb, dim_noise, encoder, data, behaviour_cl_enc, date_feature, xiP_array,\
                idx_array, client_info, cont_features, X_oh, scaler_emb, scaler_cl_emb, scaler_cont,\
                label_encoders, decoder_cl_emb, decoder_onehot, dim_Xcl, dim_X_cont, 'CBNormalize',
                'PATEGAN', eps, load, DEVICE, DIRECTORY, experiment_id)

    return data, synth_df, synth_df_rdp, synth_df_ae, synth_df_dpgan, synth_df_trgan_dpgan, synth_df_pategan
