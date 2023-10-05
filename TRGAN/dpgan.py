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
from scipy.stats import wasserstein_distance, entropy
# from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from rdt.transformers.numerical import ClusterBasedNormalizer
from rdt.transformers.categorical import FrequencyEncoder
# import scipy.stats as sts
from scipy import signal
from functorch import vmap
# from torch import vmap

from TRGAN.encoders import *
import TRGAN.Privacy_modules as privacy_trgan
from TRGAN.TRGAN_main import *

def train_dpgan(X_emb, cond_vector, dim_Vc, dim_X_emb, dim_noise=5, batch_size=2**9, lr_rates=[3e-4, 3e-4, 3e-4, 3e-4],\
                    num_epochs=15, num_blocks_gen=1, num_blocks_dis=2, h_dim=2**7, lambda1=3, alpha=0.75, window_size=25, device='cpu',\
                    privacy_type='DPGAN', eps=1e-3):

    data_dim = dim_X_emb
    z_dim = dim_noise + dim_Vc
    gauss_filter_dim = window_size

    generator = privacy_trgan.Generator(z_dim, data_dim, h_dim, num_blocks_gen, gauss_filter_dim, device).to(device)
    discriminator = privacy_trgan.Discriminator(data_dim + dim_Vc, h_dim, num_blocks_dis, gauss_filter_dim, device).to(device)
    supervisor = privacy_trgan.Supervisor(data_dim + dim_Vc, data_dim, h_dim, num_blocks_gen, gauss_filter_dim, device).to(device)
    discriminator2 = privacy_trgan.Discriminator(data_dim + dim_Vc, h_dim, num_blocks_dis, gauss_filter_dim, device).to(device)

    optimizer_G = optim.Adam(generator.parameters(), lr=lr_rates[0], betas=(0.9, 0.999), amsgrad=True)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr_rates[1], betas=(0.9, 0.999), amsgrad=True)
    optimizer_S = optim.Adam(supervisor.parameters(), lr=lr_rates[2], betas=(0.9, 0.999), amsgrad=True)
    optimizer_D2 = optim.Adam(discriminator2.parameters(), lr=lr_rates[3], betas=(0.9, 0.999), amsgrad=True)

    data_with_cv = torch.cat([torch.FloatTensor(X_emb), torch.FloatTensor(cond_vector)], axis=1)
    loader_g = DataLoader(data_with_cv, batch_size=batch_size, shuffle=True)

    epochs = tqdm(range(num_epochs))
    loss_array = []

    b_d1 = 0.01
    b_d2 = 0.01
    q = batch_size / len(X_emb)

    if privacy_type == 'TRGAN_DPGAN':
        delta = 0.1 #or 1/batch_size
        n_iter = len(X_emb) // batch_size
        epsilon_bar_array = []

        print(f'TRGAN with ({eps}, {delta})-differential privacy')


        for epoch in epochs:
            for batch_idx, X in enumerate(loader_g):
                loss = nn.MSELoss()
                batch_size = X.size(0)

                Vc = X[:, -dim_Vc:].to(device)
                
                noise = torch.randn(batch_size, dim_noise).to(device)
                z = torch.cat([noise, Vc], dim=1).to(device)
                
                fake = generator(z).detach()
                X = X.to(device)
                
                discriminator.trainable = True
                
                disc_loss = (-torch.mean(discriminator(X)) + torch.mean(discriminator(torch.cat([fake, Vc], dim=1)))).to(device)

                fake_super = supervisor(torch.cat([fake, Vc], dim=1)).to(device)
                disc2_loss = (-torch.mean(discriminator2(X)) + torch.mean(discriminator2(torch.cat([fake_super, Vc], dim=1)))).to(device)

                
                for param in discriminator.parameters():
                    if param.grad != None:
                        normed_grad = torch.div(param.grad, torch.maximum(torch.FloatTensor([1]), torch.linalg.norm(param.grad, 2)))
                        noise_grad = normed_grad + torch.normal(torch.zeros(normed_grad.size()), torch.FloatTensor([2 * q * np.sqrt(n_iter * np.log(1/delta)) / eps]))
                        param.grad = noise_grad
                
                for param in discriminator2.parameters():
                    if param.grad != None:
                        normed_grad = torch.div(param.grad, torch.maximum(torch.FloatTensor([1]), torch.linalg.norm(param.grad, 2)))
                        noise_grad = normed_grad + torch.normal(torch.zeros(normed_grad.size()), torch.FloatTensor([2 * q * np.sqrt(n_iter * np.log(1/delta)) / eps]))
                        param.grad = noise_grad
                
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

                    gen_loss = (alpha * gen_loss1 + (1 - alpha) * supervisor_loss)
                    
                    supervisor_loss2 = ((-torch.mean(discriminator2(torch.cat([supervisor(torch.cat([generator(z), Vc], dim=1).detach()),\
                        Vc], dim=1)))) + lambda1 * loss(supervisor(torch.cat([generator(z), Vc], dim=1).detach()), X[:,:-dim_Vc])).to(device)
                    
                    
                    optimizer_G.zero_grad()
                    gen_loss.backward()
                    optimizer_G.step()
                    
                    optimizer_S.zero_grad()
                    supervisor_loss2.backward()
                    optimizer_S.step()


            epsilon_bar_array.append(eps)

            epochs.set_description('Discriminator Loss: %.5f || Discriminator 2 Loss: %.5f || Generator Loss: %.5f || Supervisor Loss: %.5f' %\
                (disc_loss.item(), disc2_loss.item(), gen_loss.item(), supervisor_loss2.item()))
            loss_array.append([disc_loss.item(), disc2_loss.item(), gen_loss.item(), supervisor_loss2.item()])

        return generator, supervisor, loss_array, discriminator, discriminator2, epsilon_bar_array
    
    elif privacy_type == 'DPGAN':
        generator = privacy_trgan.Generator(dim_noise, data_dim, h_dim, num_blocks_gen, gauss_filter_dim, device).to(device)
        discriminator = privacy_trgan.Discriminator(data_dim, h_dim, num_blocks_dis, gauss_filter_dim, device).to(device)

        delta = 0.1 #or 1/batch_size
        n_iter = len(X_emb) // batch_size
        epsilon_bar_array = []

        print(f'TRGAN with ({eps}, {delta})-differential privacy')


        for epoch in epochs:
            for batch_idx, X in enumerate(loader_g):
                loss = nn.MSELoss()
                batch_size = X.size(0)

                noise = torch.randn(batch_size, dim_noise).to(device)
                z = noise
                
                fake = generator(z).detach()
                X = X[:,:-dim_Vc].to(device)
                
                discriminator.trainable = True
                
                disc_loss = (-torch.mean(discriminator(X)) + torch.mean(discriminator(fake))).to(device)

                for param in discriminator.parameters():
                    if param.grad != None:
                        normed_grad = torch.div(param.grad, torch.maximum(torch.FloatTensor([1]), torch.linalg.norm(param.grad, 2)))
                        noise_grad = normed_grad + torch.normal(torch.zeros(normed_grad.size()), torch.FloatTensor([2 * q * np.sqrt(n_iter * np.log(1/delta)) / eps]))
                        param.grad = noise_grad
                
                optimizer_D.zero_grad()
                disc_loss.backward()
                optimizer_D.step()

                for dp in discriminator.parameters():
                            dp.data.clamp_(-b_d1, b_d1)

                if batch_idx % 2 == 0:
                    discriminator.trainable = False
                    gen_loss = -torch.mean(discriminator(generator(z))).to(device)
          
                    optimizer_G.zero_grad()
                    gen_loss.backward()
                    optimizer_G.step()
                    
            epsilon_bar_array.append(eps)

            epochs.set_description('Discriminator Loss: %.5f || Generator Loss: %.5f' % (disc_loss.item(), gen_loss.item()))
            loss_array.append([disc_loss.item(), gen_loss.item()])

        return generator, loss_array, discriminator, epsilon_bar_array
    
    elif privacy_type == 'PATEGAN':
        generator = privacy_trgan.Generator(dim_noise, data_dim, h_dim, num_blocks_gen, gauss_filter_dim, device).to(device)

        teachers = []
        teachers_optim = []

        for i in range(6):
            disc = privacy_trgan.Discriminator_pategan(data_dim, h_dim, num_blocks_dis, gauss_filter_dim, device).to(device)
            teachers.append(disc)
            teachers_optim.append(optim.Adam(disc.parameters(), lr=lr_rates[1], betas=(0.9, 0.999), amsgrad=True))

        classifier = privacy_trgan.Discriminator_pategan(data_dim, h_dim, num_blocks_dis, gauss_filter_dim, device).to(device)
        optim_c = optim.Adam(classifier.parameters(), lr=lr_rates[1], betas=(0.9, 0.999), amsgrad=True)

        delta = 0.1 #or 1/batch_size
        n_iter = len(X_emb) // batch_size
        epsilon_bar_array = []

        # print(f'PATEGAN with ({eps}, {delta})-differential privacy')


        for epoch in epochs:
            for batch_idx, X in enumerate(loader_g):
                loss = nn.BCELoss()
                batch_size = X.size(0)

                noise = torch.randn(batch_size, dim_noise).to(device)
                z = noise
                
                X = X[:,:-dim_Vc].to(device)

                labels = []

                for i in range(len(teachers)):
                    
                    part_length = len(X)//6
                    z_teacher = torch.randn(part_length, dim_noise).to(device)
                    fake = generator(z_teacher).detach()

                    teacher_loss = -(torch.mean(torch.log(teachers[i](X[part_length*i : part_length*(i+1),:]) + 1e-8) + torch.log(1 - teachers[i](fake) + 1e-8))).to(device)

                    labels.append(torch.where(teachers[i](fake) < 0.5, 0, 1).detach().cpu().numpy().reshape(1, -1)[0])

                    teachers_optim[i].zero_grad()
                    teacher_loss.backward()
                    teachers_optim[i].step()

                    for dp in teachers[i].parameters():
                            dp.data.clamp_(-b_d1, b_d1)

                # print(labels)
                r = torch.where((torch.sum(torch.FloatTensor(labels), 0) + torch.FloatTensor(np.random.laplace(1/eps, size=(len(labels[0]))))) < 2.5, 0, 1)
                # print(r)

                classifier_loss = torch.mean(r * torch.log(classifier(generator(z) + 1e-8) + (1 - r) * torch.log(1 - classifier(generator(z)) + 1e-8)))

                optim_c.zero_grad()
                classifier_loss.backward()
                optim_c.step()

                for dp in classifier.parameters():
                            dp.data.clamp_(-b_d1, b_d1)


                gen_loss = torch.mean(torch.log(1 - classifier(generator(z)) + 1e-8)).to(device)
        
                optimizer_G.zero_grad()
                gen_loss.backward()
                optimizer_G.step()
                    
            epsilon_bar_array.append(eps)

            epochs.set_description('Student Loss: %.5f || Generator Loss: %.5f' % (classifier_loss.item(), gen_loss.item()))
            loss_array.append([classifier_loss.item(), gen_loss.item()])

        return generator, loss_array, discriminator, epsilon_bar_array
    

def sample_trgan_dpgan(X_emb, cond_vector, dim_Vc, dim_X_emb, dim_noise, encoder, data, behaviour_cl_enc, date_feature, xiP_array,\
                idx_array, client_info, cont_features, X_oh, scaler_emb, scaler_cl_emb, scaler_cont,\
                label_encoders, decoder_cl_emb, decoder_onehot, dim_Xcl, dim_X_cont, type_scale_cont,
                privacy_type, eps, load, DEVICE, DIRECTORY, experiment_id):
    
    h_dim = 2**6
    num_blocks_gen = 1
    num_blocks_dis = 1
    gauss_filter_dim = 20

    if load:
        generator = privacy_trgan.Generator(dim_noise + dim_Vc, dim_X_emb, h_dim, num_blocks_gen, gauss_filter_dim, DEVICE).to(DEVICE)
        supervisor = privacy_trgan.Supervisor(dim_X_emb + dim_Vc, dim_X_emb, h_dim, num_blocks_gen, gauss_filter_dim, DEVICE).to(DEVICE)

        generator.load_state_dict(torch.load(f'{DIRECTORY}{privacy_type}_{experiment_id}.pkl'))
        supervisor.load_state_dict(torch.load(f'{DIRECTORY}{privacy_type}_{experiment_id}.pkl'))

        generator.eval()
        supervisor.eval()

        loss_array = np.load(f'{DIRECTORY}loss_array_{privacy_type}_{experiment_id}.npy')

    else:
        generator, supervisor, loss_array, discriminator, discriminator2, epsilon_bar_array = train_dpgan(X_emb, cond_vector,\
                dim_Vc, dim_X_emb, dim_noise, batch_size=2**8, lr_rates=[3e-4, 3e-4, 3e-4, 3e-4], num_epochs=30, num_blocks_gen=num_blocks_gen,\
                num_blocks_dis=num_blocks_dis, h_dim=h_dim, lambda1=3, alpha=0.7, window_size=gauss_filter_dim, device=DEVICE, privacy_type=privacy_type, eps=eps)
        
        torch.save(generator.state_dict(), f'{DIRECTORY}{privacy_type}_{experiment_id}.pkl')
        torch.save(supervisor.state_dict(), f'{DIRECTORY}{privacy_type}_{experiment_id}.pkl')

        np.save(f'{DIRECTORY}loss_array_{privacy_type}_{experiment_id}.npy', loss_array)

        generator.eval()
        supervisor.eval()

    n_samples = len(X_emb)
    synth_data, synth_time = sample(n_samples, generator, supervisor, dim_noise, cond_vector, X_emb, encoder, data, behaviour_cl_enc,\
                                date_feature, 'customer', time='initial', model_time='poisson', n_splits=4, opt_time=False,\
                                xi_array=xiP_array, q_array=idx_array, device=DEVICE)
    
    synth_df, synth_df_cat = inverse_transforms(n_samples, synth_data, synth_time, client_info, cont_features, X_oh, scaler_emb, scaler_cl_emb, scaler_cont,\
                    label_encoders, decoder_cl_emb, decoder_onehot, dim_Xcl, dim_X_cont, type_scale_cont=type_scale_cont, device=DEVICE)

    return loss_array, synth_df


def sample_dpgan(X_emb, cond_vector, dim_Vc, dim_X_emb, dim_noise, encoder, data, behaviour_cl_enc, date_feature, xiP_array,\
                idx_array, client_info, cont_features, X_oh, scaler_emb, scaler_cl_emb, scaler_cont,\
                label_encoders, decoder_cl_emb, decoder_onehot, dim_Xcl, dim_X_cont, type_scale_cont,
                privacy_type, eps, load, DEVICE, DIRECTORY, experiment_id):
    
    h_dim = 2**6
    num_blocks_gen = 1
    num_blocks_dis = 1
    gauss_filter_dim = 20

    if load:
        generator = privacy_trgan.Generator(dim_noise, dim_X_emb, h_dim, num_blocks_gen, gauss_filter_dim, DEVICE).to(DEVICE)
        generator.load_state_dict(torch.load(f'{DIRECTORY}{privacy_type}_{experiment_id}.pkl'))
        generator.eval()

        loss_array = np.load(f'{DIRECTORY}loss_array_{privacy_type}_{experiment_id}.npy')

    else:
        generator, loss_array, discriminator, epsilon_bar_array = train_dpgan(X_emb, cond_vector,\
                dim_Vc, dim_X_emb, dim_noise, batch_size=2**8, lr_rates=[3e-4, 3e-4, 3e-4, 3e-4], num_epochs=30, num_blocks_gen=num_blocks_gen,\
                num_blocks_dis=num_blocks_dis, h_dim=h_dim, lambda1=3, alpha=0.7, window_size=gauss_filter_dim, device=DEVICE, privacy_type=privacy_type, eps=eps)
        
        torch.save(generator.state_dict(), f'{DIRECTORY}{privacy_type}_{experiment_id}.pkl')

        np.save(f'{DIRECTORY}loss_array_{privacy_type}_{experiment_id}.npy', loss_array)
        generator.eval()

    n_samples = len(X_emb)
    synth_data, synth_time = sample_data_dpgan(n_samples, generator, dim_noise, cond_vector, X_emb, encoder, data, behaviour_cl_enc,\
                                date_feature, 'customer', time='initial', model_time='poisson', n_splits=4, opt_time=False,\
                                xi_array=xiP_array, q_array=idx_array, device=DEVICE)
    
    synth_df, synth_df_cat = inverse_transforms(n_samples, synth_data, synth_time, client_info, cont_features, X_oh, scaler_emb, scaler_cl_emb, scaler_cont,\
                    label_encoders, decoder_cl_emb, decoder_onehot, dim_Xcl, dim_X_cont, type_scale_cont=type_scale_cont, device=DEVICE)

    return loss_array, synth_df

def sample_data_dpgan(n_samples, generator, noise_dim, cond_vector, X_emb, encoder, data, behaviour_cl_enc,\
            date_feature, name_client_id, time='initial', model_time='poisson', n_splits=2, opt_time=True, xi_array=[], q_array=[], device='cpu'):
    if n_samples <= len(cond_vector):
        
        X_emb_cv = encoder(torch.FloatTensor(X_emb).to(device)).detach().cpu().numpy()
        synth_time, cond_vector = sample_cond_vector_with_time(n_samples, len(cond_vector), X_emb_cv, data,\
                        behaviour_cl_enc, date_feature, name_client_id, time, model_time, n_splits, opt_time, xi_array, q_array)
        
        noise = torch.randn(n_samples, noise_dim)
        synth_data = generator(noise).detach().cpu().numpy()
        synth_time = synth_time[:n_samples]
    # synth_time = data[date_feature]

    else:
        X_emb = encoder(torch.FloatTensor(X_emb).to(device)).detach().cpu().numpy()
        synth_time, cond_vector = sample_cond_vector_with_time(n_samples, len(cond_vector), X_emb, data,\
                        behaviour_cl_enc, date_feature, name_client_id, time, model_time, n_splits, opt_time, xi_array, q_array)
        
        noise = torch.randn(n_samples, noise_dim)
        synth_data = generator(noise).detach().cpu().numpy()

    synth_time = synth_time.sort_values(by='transaction_date')

    return synth_data, synth_time