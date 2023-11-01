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


        #first layer
        self.fc1 = nn.Linear(self.z_dim, self.h_dim)
        self.relu = nn.LeakyReLU(0.2)
        self.lrelu = nn.LeakyReLU(0.2)
        self.layernorm0 = nn.LayerNorm(self.h_dim, elementwise_affine=True)
        # self.layernorm0 = nn.BatchNorm1d(self.h_dim, affine=True)
        

        #convolution layers
        self.linear_layers_conv1 = nn.ModuleList([nn.Linear(self.h_dim, 2**6) for _ in range(self.num_blocks)])
        self.linear_layers_conv2 = nn.ModuleList([nn.Linear(self.h_dim, 2**6) for _ in range(self.num_blocks)])
        self.linear_layers_conv3 = nn.ModuleList([nn.Linear(self.h_dim, 2**6) for _ in range(self.num_blocks)])
        self.linear_layers = nn.ModuleList([nn.Linear(3 * 2**6, self.h_dim) for _ in range(self.num_blocks)])
        self.dropout = nn.Dropout(0.3)

        self.layernorm_layers_1 = nn.ModuleList([nn.LayerNorm(self.h_dim, elementwise_affine=True) for _ in range(self.num_blocks)])
        self.layernorm_layers_2 = nn.ModuleList([nn.LayerNorm(self.h_dim, elementwise_affine=True) for _ in range(self.num_blocks)])
        # self.layernorm_layers_1 = nn.ModuleList([nn.BatchNorm1d(self.h_dim, affine=True) for _ in range(self.num_blocks)])
        # self.layernorm_layers_2 = nn.ModuleList([nn.BatchNorm1d(self.h_dim, affine=True) for _ in range(self.num_blocks)])

        #last layer
        self.fc2 = nn.Linear(self.h_dim, self.data_dim)
        self.tanh = nn.Tanh()
        

        # self.feed_forward_generator_layers = nn.ModuleList(
        #     [nn.Sequential(
        #     nn.Linear(self.h_dim, self.h_dim),
        #     nn.ReLU(),
        #     nn.Dropout(0.2),
        #     nn.Linear(self.h_dim, self.h_dim)
        # ) for _ in range(self.num_blocks)]
        # )

        self.feed_forward_generator_layers2 = nn.ModuleList(
            [nn.Sequential(
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
        ) for _ in range(self.num_blocks)]
        )


        self.filter1 = signal.windows.gaussian(gauss_filter_dim, 0.5)
        self.filter2 = signal.windows.gaussian(gauss_filter_dim, 1)
        self.filter3 = signal.windows.gaussian(gauss_filter_dim, 3)

        # self.init_weights()
        # self.linear_layers.apply(init_weight_seq)
        # self.linear_layers_conv1.apply(init_weight_seq)
        # self.linear_layers_conv2.apply(init_weight_seq)
        # self.linear_layers_conv3.apply(init_weight_seq)
        
    # def init_weights(self):
    #     torch.nn.init.normal_(self.fc1.weight, 0, 0.02)
    #     torch.nn.init.normal_(self.fc2.weight, 0, 0.02)

            
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
            # out = self.dropout(out)

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


        #first layer
        self.fc1 = nn.Linear(self.z_dim, self.h_dim)
        self.relu = nn.LeakyReLU(0.2)
        self.lrelu = nn.LeakyReLU(0.2)

        self.layernorm0 = nn.LayerNorm(self.h_dim, elementwise_affine=True)
        # self.layernorm0 = nn.BatchNorm1d(self.h_dim, affine=True)


        #convolution layers
        self.linear_layers_conv1 = nn.ModuleList([nn.Linear(self.h_dim, 2**6) for _ in range(self.num_blocks)])
        self.linear_layers_conv2 = nn.ModuleList([nn.Linear(self.h_dim, 2**6) for _ in range(self.num_blocks)])
        self.linear_layers_conv3 = nn.ModuleList([nn.Linear(self.h_dim, 2**6) for _ in range(self.num_blocks)])
        self.linear_layers = nn.ModuleList([nn.Linear(3 * 2**6, self.h_dim) for _ in range(self.num_blocks)])
        self.dropout = nn.Dropout(0.3)

        # self.feed_forward_generator_layers = nn.ModuleList(
        #     [nn.Sequential(
        #     nn.Linear(self.h_dim, self.h_dim),
        #     nn.ReLU(),
        #     nn.Dropout(0.2),
        #     nn.Linear(self.h_dim, self.h_dim)
        # ) for _ in range(self.num_blocks)]
        # )

        self.feed_forward_generator_layers2 = nn.ModuleList(
            [nn.Sequential(
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
        ) for _ in range(self.num_blocks)]
        )

        self.layernorm_layers_1 = nn.ModuleList([nn.LayerNorm(self.h_dim, elementwise_affine=True) for _ in range(self.num_blocks)])
        self.layernorm_layers_2 = nn.ModuleList([nn.LayerNorm(self.h_dim, elementwise_affine=True) for _ in range(self.num_blocks)])
        # self.layernorm_layers_1 = nn.ModuleList([nn.BatchNorm1d(self.h_dim, affine=True) for _ in range(self.num_blocks)])
        # self.layernorm_layers_2 = nn.ModuleList([nn.BatchNorm1d(self.h_dim, affine=True) for _ in range(self.num_blocks)])

        self.filter1 = signal.windows.gaussian(gauss_filter_dim, 0.5)
        self.filter2 = signal.windows.gaussian(gauss_filter_dim, 1)
        self.filter3 = signal.windows.gaussian(gauss_filter_dim, 3)


        #last layer
        self.fc2 = nn.Linear(self.h_dim, self.data_dim)
        self.tanh = nn.Tanh()

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
            # out = self.dropout(out)

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


        #first layer
        self.fc1 = nn.Linear(self.data_dim, self.h_dim)
        self.lrelu = nn.LeakyReLU(0.1)
        self.relu = nn.LeakyReLU(0.1)

        self.layernorm0 = nn.LayerNorm(self.h_dim, elementwise_affine=True)
        # self.layernorm0 = nn.BatchNorm1d(self.h_dim, affine=True)

        
        #convolution layer
        self.linear_layers_conv1 = nn.ModuleList([nn.Linear(self.h_dim, 2**6) for _ in range(self.num_blocks)])
        self.linear_layers_conv2 = nn.ModuleList([nn.Linear(self.h_dim, 2**6) for _ in range(self.num_blocks)])
        self.linear_layers_conv3 = nn.ModuleList([nn.Linear(self.h_dim, 2**6) for _ in range(self.num_blocks)])
        self.linear_layers = nn.ModuleList([nn.Linear(3 * 2**6, self.h_dim) for _ in range(self.num_blocks)])
        self.dropout = nn.Dropout(0.2)

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

        self.layernorm_layers_1 = nn.ModuleList([nn.LayerNorm(self.h_dim, elementwise_affine=True) for _ in range(self.num_blocks)])
        self.layernorm_layers_2 = nn.ModuleList([nn.LayerNorm(self.h_dim, elementwise_affine=True) for _ in range(self.num_blocks)])
        # self.layernorm_layers_1 = nn.ModuleList([nn.BatchNorm1d(self.h_dim, affine=True) for _ in range(self.num_blocks)])
        # self.layernorm_layers_2 = nn.ModuleList([nn.BatchNorm1d(self.h_dim, affine=True) for _ in range(self.num_blocks)])

        self.filter1 = signal.windows.gaussian(gauss_filter_dim, 0.5)
        self.filter2 = signal.windows.gaussian(gauss_filter_dim, 1)
        self.filter3 = signal.windows.gaussian(gauss_filter_dim, 3)


        #last layer
        self.fc2 = nn.Linear(self.h_dim, 1)
        self.tanh = nn.Tanh()

        # self.init_weights()
        # self.linear_layers.apply(init_weight_seq)
        # self.linear_layers_conv1.apply(init_weight_seq)
        # self.linear_layers_conv2.apply(init_weight_seq)
        # self.linear_layers_conv3.apply(init_weight_seq)

    # def init_weights(self):
    #     torch.nn.init.normal_(self.fc1.weight, 0, 0.02)
    #     torch.nn.init.normal_(self.fc2.weight, 0, 0.02)
    
            
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
            # out = self.dropout(out)

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
        # return self.tanh(out)
        return out
    


class Discriminator_pategan(nn.Module):
    def __init__(self, data_dim, h_dim, num_blocks, gauss_filter_dim, device):
        super(Discriminator_pategan, self).__init__()
        
        self.data_dim = data_dim
        self.h_dim = h_dim
        self.num_blocks = num_blocks
        self.gauss_filter_dim = gauss_filter_dim
        self.device = device

        #first layer
        self.fc1 = nn.Linear(self.data_dim, self.h_dim)
        self.lrelu = nn.LeakyReLU(0.1)
        self.relu = nn.LeakyReLU(0.1)

        self.layernorm0 = nn.LayerNorm(self.h_dim, elementwise_affine=True)

        self.feed_forward_discriminator_layers2 = nn.ModuleList(
            [nn.Sequential(
            nn.Linear(self.h_dim, self.h_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(self.h_dim, self.h_dim),
            nn.LeakyReLU(0.1)
        ) for _ in range(self.num_blocks)]
        )

        self.layernorm_layers_1 = nn.ModuleList([nn.LayerNorm(self.h_dim, elementwise_affine=True) for _ in range(self.num_blocks)])
        self.layernorm_layers_2 = nn.ModuleList([nn.LayerNorm(self.h_dim, elementwise_affine=True) for _ in range(self.num_blocks)])
 
        #last layer
        self.fc2 = nn.Linear(self.h_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_size = x.size()
        out = self.layernorm0(self.relu(self.fc1(x)))

        for i in range(self.num_blocks):
            res = out

            out = self.feed_forward_discriminator_layers2[i](out)

            #add & norm
            out += res
            out = self.layernorm_layers_1[i](out)
  
        out = self.fc2(out)
        return self.sigmoid(out)
    


    
class PrivacyOperator(nn.Module):
    def __init__(self, data_dim):
        super(PrivacyOperator, self).__init__()

        self.data_dim = data_dim

        self.model = nn.Sequential(
            nn.Linear(self.data_dim, 2**7),
            nn.ReLU(),
            nn.Linear(2**7, 2**7),
            nn.ReLU(),
            nn.Linear(2**7, 1)
            # nn.Sigmoid()
        )

    def forward(self, x):
        out = self.model(x)
        return out


# def train_generator2(X_emb, cond_vector, dim_Vc, dim_X_emb, dim_noise=5, batch_size=2**9, lr_rates=[3e-4, 3e-4, 3e-4, 3e-4],\
#                      num_epochs=15, num_blocks_gen=1, num_blocks_dis=2, h_dim=2**7, privacy_type='RDP', epsilon=0.01):
#     # date_transf_dim = num_date_features
#     # pos_dim = behaviour_cl_enc.shape[1]
#     data_dim = dim_X_emb
#     z_dim = dim_noise + dim_Vc

#     generator = Generator(z_dim, data_dim, h_dim, num_blocks_gen).to(device)
#     discriminator = Discriminator(data_dim + dim_Vc, h_dim, num_blocks_dis).to(device)
#     supervisor = Supervisor(data_dim + dim_Vc, data_dim, h_dim, num_blocks_gen).to(device)
#     discriminator2 = Discriminator(data_dim + dim_Vc, h_dim, num_blocks_dis).to(device)
#     privacyblock = PrivacyOperator(data_dim).to(device)

#     optimizer_G = optim.Adam(generator.parameters(), lr=lr_rates[0], betas=(0.9, 0.999))
#     optimizer_D = optim.Adam(discriminator.parameters(), lr=lr_rates[1], betas=(0.9, 0.999))
#     optimizer_S = optim.Adam(supervisor.parameters(), lr=lr_rates[2], betas=(0.9, 0.999))
#     optimizer_D2 = optim.Adam(discriminator2.parameters(), lr=lr_rates[3], betas=(0.9, 0.999))
#     optimizer_privacy = optim.Adam(privacyblock.parameters(), lr=3e-4)

#     # optimizer_G = optim.RMSprop(generator.parameters(), lr=lr_rates[0])
#     # optimizer_D = optim.RMSprop(discriminator.parameters(), lr=lr_rates[1])
#     # optimizer_S = optim.RMSprop(supervisor.parameters(), lr=lr_rates[2])
#     # optimizer_D2 = optim.RMSprop(discriminator2.parameters(), lr=lr_rates[3])

#     scheduler_G = torch.optim.lr_scheduler.ExponentialLR(optimizer_G, gamma=0.97)
#     scheduler_D = torch.optim.lr_scheduler.ExponentialLR(optimizer_D, gamma=0.97)
#     scheduler_S = torch.optim.lr_scheduler.ExponentialLR(optimizer_S, gamma=0.97)
#     scheduler_D2 = torch.optim.lr_scheduler.ExponentialLR(optimizer_D2, gamma=0.97)

#     data_with_cv = torch.cat([torch.FloatTensor(X_emb), torch.FloatTensor(cond_vector)], axis=1)

#     idx_batch_array = np.arange(len(data_with_cv)//batch_size * batch_size)
#     last_idx = np.setdiff1d(np.arange(len(data_with_cv)), idx_batch_array)
#     split_idx = np.split(idx_batch_array, batch_size)
#     split_idx_perm = np.random.permutation(split_idx)
#     split_idx_perm = np.append(split_idx_perm, last_idx)

#     loader_g = DataLoader(data_with_cv[split_idx_perm], batch_size=batch_size, shuffle=False)

#     epochs = tqdm(range(num_epochs))
    # loss_array = []

    # b_d1 = 0.015
    # b_d2 = 0.02
    # q = batch_size / len(X_emb)

    # if privacy_type == 'RDP':
    #     alpha = 1.1
    #     delta = 0.2
    #     sensitivity = 2*4*(5*b_d1*(h_dim + 1))/batch_size
    #     n_iter = len(X_emb) // batch_size
        
    #     print(f'TRGAN with ({epsilon}, {delta})-differential privacy')
    #     epsilon_bar_array = []

    #     std = (2 * q * alpha**2 * sensitivity**2 * np.sqrt(2 * n_iter * np.log(1/delta))) / (2 * (alpha-1) * epsilon)

    #     for epoch in epochs:
    #         for batch_idx, X in enumerate(loader_g):
    #             loss = torch.nn.MSELoss()

    #             batch_size = X.size(0)
    #             X = X.to(device)

    #             Vc = X[:, -dim_Vc:].to(device)
                
    #             # noise = torch.randn(batch_size, dim_noise).to(device)
    #             noise = torch.FloatTensor(dclProcess(batch_size - 1, dim_noise)).to(device)
    #             z = torch.cat([noise, Vc], dim=1).to(device)
    #             privacy_noise = torch.FloatTensor(np.random.normal(0, std, size=(10, 1))).to(device)
                
    #             fake = generator(z).detach()
    #             disc_loss = (-torch.mean(discriminator(X[:,:])) + torch.mean(discriminator(torch.cat([fake, Vc], dim=1)))).to(device)

    #             fake_super = supervisor(torch.cat([fake, Vc], dim=1)).to(device)
    #             disc2_loss = (-torch.mean(discriminator2(X) + privacy_noise) + torch.mean(discriminator2(torch.cat([fake_super, Vc], dim=1)) + privacy_noise)).to(device) 
            

    #             for dp in discriminator.parameters():
    #                         dp.data.clamp_(-b_d1, b_d1)

    #             for dp in discriminator2.parameters():
    #                         dp.data.clamp_(-b_d2, b_d2)
                

    #             optimizer_D.zero_grad()
    #             disc_loss.backward()
    #             optimizer_D.step()

    #             optimizer_D2.zero_grad()
    #             disc2_loss.backward()
    #             optimizer_D2.step()
                        
    #             if batch_idx % 5 == 0:
    #                 gen_loss1 = -torch.mean(discriminator(torch.cat([generator(z), Vc], dim=1))).to(device)
    #                 supervisor_loss = (-torch.mean(discriminator2(torch.cat([supervisor(torch.cat([generator(z), Vc], dim=1).detach()), Vc], dim=1)))).to(device)
    #                                 #    + loss(supervisor(torch.cat([generator(z), Vc], dim=1).detach()), X[:,:-dim_Vc])).to(device)

    #                 gen_loss = (0.7*gen_loss1 + 0.3*supervisor_loss)
                
    #                 optimizer_G.zero_grad()
    #                 gen_loss.backward()
    #                 optimizer_G.step()
                    
    #                 supervisor_loss2 = ((-torch.mean(discriminator2(torch.cat([supervisor(torch.cat([generator(z), Vc], dim=1).detach()),\
    #                     Vc], dim=1)))) + 3*loss(supervisor(torch.cat([generator(z), Vc], dim=1).detach()), X[:,:-dim_Vc])).to(device)
                    
                    
    #                 optimizer_S.zero_grad()
    #                 supervisor_loss2.backward()
    #                 optimizer_S.step()
            
    #         epsilon_bar_array.append(epsilon)

    #         scheduler_G.step()
    #         scheduler_D.step()
    #         scheduler_S.step()
    #         scheduler_D2.step()

    #         epochs.set_description('Discriminator Loss: %.5f || Discriminator 2 Loss: %.5f || Generator Loss: %.5f || Supervisor Loss: %.5f' %\
    #             (disc_loss.item(), disc2_loss.item(), gen_loss.item(), supervisor_loss2.item()))
    #         loss_array.append([disc_loss.item(), disc2_loss.item(), gen_loss.item(), supervisor_loss2.item()])
    
    # elif privacy_type == 'SDP_d':
    #     alpha = 1.1
    #     delta = 0.2
    #     sensitivity = 2*4*(5*b_d1*(h_dim + 1))/batch_size
    #     n_iter = len(X_emb) // batch_size
    #     epsilon_0 = 0.01
        
    #     # print(f'TRGAN with ({epsilon}, {delta})-differential privacy')
    #     epsilon_bar_array = []

    #     for epoch in epochs:
    #         for batch_idx, X in enumerate(loader_g):
    #             loss = torch.nn.MSELoss()
    #             loss_privacy_mse = torch.nn.MSELoss()

    #             batch_size = X.size(0)
    #             X = X.to(device)

    #             Vc = X[:, -dim_Vc:].to(device)
                
    #             # noise = torch.randn(batch_size, dim_noise).to(device)
    #             noise = torch.FloatTensor(dclProcess(batch_size - 1, dim_noise)).to(device)
    #             z = torch.cat([noise, Vc], dim=1).to(device)
                
    #             fake = generator(z).detach()
                
    #             epsilon = privacyblock(fake.to(device)).to(device)
    #             epsilon_bar = torch.mean(epsilon)

    #             loss_privacy = loss_privacy_mse(epsilon_bar, torch.FloatTensor([epsilon_0]).to(device))
    #             # privacy_noise = torch.FloatTensor([np.repeat(np.random.normal(0, 2*np.log(1.25/delta)*(sensitivity**2)/(eps**2)), data_dim) for eps in epsilon.detach().cpu().numpy()]).to(device)
    #             privacy_noise = torch.FloatTensor([np.random.normal(0, 4*(q * alpha**2 * sensitivity/batch_size) / (2*(alpha-1)*eps) * np.sqrt(2 * n_iter * np.log(1/delta))) for eps in epsilon.detach().cpu().numpy()]).to(device)

    #             disc_loss = (-torch.mean(discriminator(X[:,:])) + torch.mean(discriminator(torch.cat([fake, Vc], dim=1)))).to(device)

    #             fake_super = supervisor(torch.cat([fake, Vc], dim=1)).to(device)
    #             disc2_loss = (-torch.mean(discriminator2(X) + privacy_noise) + torch.mean(discriminator2(torch.cat([fake_super, Vc], dim=1)) + privacy_noise)).to(device) 
            

    #             for dp in discriminator.parameters():
    #                         dp.data.clamp_(-b_d1, b_d1)

    #             for dp in discriminator2.parameters():
    #                         dp.data.clamp_(-b_d2, b_d2)
                

    #             optimizer_D.zero_grad()
    #             disc_loss.backward()
    #             optimizer_D.step()

    #             optimizer_D2.zero_grad()
    #             disc2_loss.backward()
    #             optimizer_D2.step()

    #             optimizer_privacy.zero_grad()
    #             loss_privacy.backward()
    #             optimizer_privacy.step()
                        
    #             if batch_idx % 5 == 0:
    #                 gen_loss1 = -torch.mean(discriminator(torch.cat([generator(z), Vc], dim=1))).to(device)
    #                 supervisor_loss = (-torch.mean(discriminator2(torch.cat([supervisor(torch.cat([generator(z), Vc], dim=1).detach()), Vc], dim=1)))).to(device)
    #                                 #    + loss(supervisor(torch.cat([generator(z), Vc], dim=1).detach()), X[:,:-dim_Vc])).to(device)

    #                 # gen_loss1 = -torch.mean(discriminator(torch.cat([fake, X[:, -(hidden_dim+date_transf_dim):]], dim=1)))
    #                 # supervisor_loss = -torch.mean(discriminator2(torch.cat([fake_super, X[:, -(hidden_dim+date_transf_dim):]], dim=1)))

    #                 gen_loss = (0.7*gen_loss1 + 0.3*supervisor_loss)
    #             #     gen_loss = torch.mean(torch.log(discriminator(torch.cat([generator(z), X[:, -(hidden_dim+date_transf_dim):]], dim=1))))
    #                 optimizer_G.zero_grad()
    #                 gen_loss.backward()
    #                 optimizer_G.step()
                    
    #                 supervisor_loss2 = ((-torch.mean(discriminator2(torch.cat([supervisor(torch.cat([generator(z), Vc], dim=1).detach()),\
    #                     Vc], dim=1)))) + 3*loss(supervisor(torch.cat([generator(z), Vc], dim=1).detach()), X[:,:-dim_Vc])).to(device)
    #                 # loss_dist(supervisor(torch.cat([generator(z), Vc], dim=1).detach()), X[:,:-dim_Vc]).to(device)
                    
    #                 # supervisor_loss2 = loss(supervisor(torch.cat([generator(z), Vc], dim=1).detach()), X[:,:-(hidden_dim+date_transf_dim+pos_dim)])
    #                 # supervisor_loss2 = -torch.mean(discriminator2(torch.cat([fake_super, X[:, -(hidden_dim+date_transf_dim):]], dim=1))) +\
    #                     #  loss(fake_super, X[:,:-(hidden_dim+date_transf_dim)])

                    
    #                 optimizer_S.zero_grad()
    #                 supervisor_loss2.backward()
    #                 optimizer_S.step()
            
    #         epsilon_bar_array.append(epsilon_bar.detach().cpu().numpy())

    #         scheduler_G.step()
    #         scheduler_D.step()
    #         scheduler_S.step()
    #         scheduler_D2.step()

    #         epochs.set_description('Discriminator Loss: %.5f || Discriminator 2 Loss: %.5f || Generator Loss: %.5f || Supervisor Loss: %.5f' %\
    #             (disc_loss.item(), disc2_loss.item(), gen_loss.item(), supervisor_loss2.item()))
    #         loss_array.append([disc_loss.item(), disc2_loss.item(), gen_loss.item(), supervisor_loss2.item()])
    #     #     print(f'epoch {epoch}: G loss = {gen_loss.item():.8f}, D loss = {disc_loss.item():.8f}')

    # elif privacy_type == 'SDP_g':
    #     alpha = 1.1
    #     # std = 0.1
    #     delta = 0.2
    #     sensitivity = 2*4*(5*b_d1*(h_dim + 1))/batch_size
    #     n_iter = len(X_emb) // batch_size
    #     epsilon_0 = 0.01
        
    #     # epsilon = 4*(q * alpha**2 * sensitivity/batch_size) / (2*(alpha-1)*std) * np.sqrt(2 * n_iter * np.log(1/delta))
    #     # print(f'TRGAN with ({epsilon}, {delta})-differential privacy')
    #     epsilon_bar_array = []
    #     # sensitivity = 2*np.sqrt(data_dim)


    #     for epoch in epochs:
    #         for batch_idx, X in enumerate(loader_g):
    #             loss = torch.nn.MSELoss()
    #             loss_privacy_mse = torch.nn.MSELoss()

    #             batch_size = X.size(0)
    #             X = X.to(device)

    #             Vc = X[:, -dim_Vc:].to(device)
                
    #             # noise = torch.randn(batch_size, dim_noise).to(device)
    #             noise = torch.FloatTensor(dclProcess(batch_size - 1, dim_noise)).to(device)
    #             z = torch.cat([noise, Vc], dim=1).to(device)
                
    #             fake = generator(z).detach()
                
    #             # epsilon = np.linspace(3, 1.5, batch_size)
    #             # epsilon_bar = torch.FloatTensor(np.mean(epsilon)).to(device)
    #             epsilon = privacyblock(fake.to(device)).to(device)
    #             epsilon_bar = torch.mean(epsilon)
    #             # loss_privacy = loss_privacy_mse(torch.cat([generator(z), Vc], dim=1).to(device), X).to(device)

    #             loss_privacy = loss_privacy_mse(epsilon_bar, torch.FloatTensor([epsilon_0]).to(device))

    #             privacy_noise = torch.FloatTensor([np.repeat(np.random.normal(0, 2*np.log(1.25/delta)*(sensitivity**2)/(eps**2)), data_dim) for eps in epsilon.detach().cpu().numpy()]).to(device)
    #             fake = fake + privacy_noise


    #             disc_loss = (-torch.mean(discriminator(X[:,:])) + torch.mean(discriminator(torch.cat([fake, Vc], dim=1)))).to(device)

    #             fake_super = supervisor(torch.cat([fake, Vc], dim=1)).to(device)
    #             disc2_loss = (-torch.mean(discriminator2(X)) + torch.mean(discriminator2(torch.cat([fake_super, Vc], dim=1)))).to(device) 
            

    #             for dp in discriminator.parameters():
    #                         dp.data.clamp_(-b_d1, b_d1)

    #             for dp in discriminator2.parameters():
    #                         dp.data.clamp_(-b_d2, b_d2)
                

    #             optimizer_D.zero_grad()
    #             disc_loss.backward()
    #             optimizer_D.step()

    #             optimizer_D2.zero_grad()
    #             disc2_loss.backward()
    #             optimizer_D2.step()

    #             optimizer_privacy.zero_grad()
    #             loss_privacy.backward()
    #             optimizer_privacy.step()
                        
    #             if batch_idx % 5 == 0:
    #                 gen_loss1 = -torch.mean(discriminator(torch.cat([generator(z), Vc], dim=1))).to(device)
    #                 supervisor_loss = (-torch.mean(discriminator2(torch.cat([supervisor(torch.cat([generator(z), Vc], dim=1).detach()), Vc], dim=1)))).to(device)
    #                                 #    + loss(supervisor(torch.cat([generator(z), Vc], dim=1).detach()), X[:,:-dim_Vc])).to(device)

    #                 # gen_loss1 = -torch.mean(discriminator(torch.cat([fake, X[:, -(hidden_dim+date_transf_dim):]], dim=1)))
    #                 # supervisor_loss = -torch.mean(discriminator2(torch.cat([fake_super, X[:, -(hidden_dim+date_transf_dim):]], dim=1)))

    #                 gen_loss = (0.7*gen_loss1 + 0.3*supervisor_loss)
    #             #     gen_loss = torch.mean(torch.log(discriminator(torch.cat([generator(z), X[:, -(hidden_dim+date_transf_dim):]], dim=1))))
    #                 optimizer_G.zero_grad()
    #                 gen_loss.backward()
    #                 optimizer_G.step()
                    
    #                 supervisor_loss2 = ((-torch.mean(discriminator2(torch.cat([supervisor(torch.cat([generator(z), Vc], dim=1).detach()),\
    #                     Vc], dim=1)))) + 3*loss(supervisor(torch.cat([generator(z), Vc], dim=1).detach()), X[:,:-dim_Vc])).to(device)
    #                 # loss_dist(supervisor(torch.cat([generator(z), Vc], dim=1).detach()), X[:,:-dim_Vc]).to(device)
                    
    #                 # supervisor_loss2 = loss(supervisor(torch.cat([generator(z), Vc], dim=1).detach()), X[:,:-(hidden_dim+date_transf_dim+pos_dim)])
    #                 # supervisor_loss2 = -torch.mean(discriminator2(torch.cat([fake_super, X[:, -(hidden_dim+date_transf_dim):]], dim=1))) +\
    #                     #  loss(fake_super, X[:,:-(hidden_dim+date_transf_dim)])

                    
    #                 optimizer_S.zero_grad()
    #                 supervisor_loss2.backward()
    #                 optimizer_S.step()
            
    #         epsilon_bar_array.append(epsilon_bar.detach().cpu().numpy())

    #         scheduler_G.step()
    #         scheduler_D.step()
    #         scheduler_S.step()
    #         scheduler_D2.step()

    #         epochs.set_description('Discriminator Loss: %.5f || Discriminator 2 Loss: %.5f || Generator Loss: %.5f || Supervisor Loss: %.5f' %\
    #             (disc_loss.item(), disc2_loss.item(), gen_loss.item(), supervisor_loss2.item()))
    #         loss_array.append([disc_loss.item(), disc2_loss.item(), gen_loss.item(), supervisor_loss2.item()])
    #     #     print(f'epoch {epoch}: G loss = {gen_loss.item():.8f}, D loss = {disc_loss.item():.8f}')


    # return generator, supervisor, loss_array, discriminator, discriminator2, epsilon_bar_array
    







def z_func(x):
    if (x <= 0) and (x >= -1):
        return x**2
    
    elif  (x <= 1) and (x > 0):
        return -(x**2)
    
    else:
        return 0
    
z_func = np.vectorize(z_func)




def train_generator(X_emb, cond_vector, dim_Vc, dim_X_emb, dim_noise=5, batch_size=2**9, lr_rates=[3e-4, 3e-4, 3e-4, 3e-4],\
                    num_epochs=15, num_blocks_gen=1, num_blocks_dis=2, h_dim=2**7, lambda1=3, alpha_r=0.75, window_size=25, device='cpu',\
                    privacy_type='RDP', eps=1e-3):

    data_dim = dim_X_emb
    z_dim = dim_noise + dim_Vc
    gauss_filter_dim = window_size

    generator = Generator(z_dim, data_dim, h_dim, num_blocks_gen, gauss_filter_dim, device).to(device)
    discriminator = Discriminator(data_dim + dim_Vc, h_dim, num_blocks_dis, gauss_filter_dim, device).to(device)
    supervisor = Supervisor(data_dim + dim_Vc, data_dim, h_dim, num_blocks_gen, gauss_filter_dim, device).to(device)
    discriminator2 = Discriminator(data_dim + dim_Vc, h_dim, num_blocks_dis, gauss_filter_dim, device).to(device)
    privacyOperator = PrivacyOperator(data_dim).to(device)

    optimizer_G = optim.Adam(generator.parameters(), lr=lr_rates[0], betas=(0.9, 0.999), amsgrad=True)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr_rates[1], betas=(0.9, 0.999), amsgrad=True)
    optimizer_S = optim.Adam(supervisor.parameters(), lr=lr_rates[2], betas=(0.9, 0.999), amsgrad=True)
    optimizer_D2 = optim.Adam(discriminator2.parameters(), lr=lr_rates[3], betas=(0.9, 0.999), amsgrad=True)
    optimizer_privacy = optim.Adam(privacyOperator.parameters(), lr=3e-4, betas=(0.9, 0.999), amsgrad=True)

    scheduler_G = torch.optim.lr_scheduler.ExponentialLR(optimizer_G, gamma=0.98)
    scheduler_D = torch.optim.lr_scheduler.ExponentialLR(optimizer_D, gamma=0.98)
    scheduler_S = torch.optim.lr_scheduler.ExponentialLR(optimizer_S, gamma=0.98)
    scheduler_D2 = torch.optim.lr_scheduler.ExponentialLR(optimizer_D2, gamma=0.98)

    data_with_cv = torch.cat([torch.FloatTensor(X_emb), torch.FloatTensor(cond_vector)], axis=1)

    idx_batch_array = np.arange(len(data_with_cv)//batch_size * batch_size)
    last_idx = np.setdiff1d(np.arange(len(data_with_cv)), idx_batch_array)
    split_idx = np.split(idx_batch_array, batch_size)
    split_idx_perm = np.random.permutation(split_idx)
    split_idx_perm = np.append(split_idx_perm, last_idx)

    loader_g = DataLoader(data_with_cv[split_idx_perm], batch_size=batch_size, shuffle=False)
    # loader_g = DataLoader(data_with_cv, batch_size=batch_size, shuffle=True)

    epochs = tqdm(range(num_epochs))
    loss_array = []

    b_d1 = 0.01
    b_d2 = 0.01
    q = batch_size / len(X_emb)

    if privacy_type == 'RDP':
        alpha = 2
        delta = 0.1 #or 1/batch_size
        C = 1
        h = X_emb.shape[1]
        k = 4
        # gamma = (C * h + 1) * (b_d1**k * h**(k-1) + b_d1) + sum([(b_d1**i * h**(i-1)) for i in range(1, k-1)])
        Sk = sum([(b_d1**i * h**(i-1)) for i in range(1, k-1)])
        gamma = 0.5 * (h-1)/(h) * b_d1 * ((b_d1**k * (C*h + 1) * h**(k-1) + Sk + b_d1 * (C*h + 1))/
                        (b_d1**(k+1) * (C*h + 1)**2 * h**(k-1) +  b_d1 * (C*h + 1) * Sk)) + 2*b_d1
        sensitivity = 4 * gamma /batch_size
        n_iter = len(X_emb) // batch_size
        
        print(f'TRGAN with ({eps}, {delta})-differential privacy')
        epsilon_bar_array = []

        #calculate the noise parameter std
        std = np.sqrt((4 * q * alpha**2 * (sensitivity)**2 * np.sqrt(2 * n_iter * np.log(1/delta))) / (2 * (alpha-1) * eps))
        

        for epoch in epochs:
            for batch_idx, X in enumerate(loader_g):
                loss = nn.MSELoss()
                batch_size = X.size(0)

                Vc = X[:, -dim_Vc:].to(device)
                
                noise = torch.FloatTensor(dclProcess(batch_size - 1, dim_noise)).to(device)
                z = torch.cat([noise, Vc], dim=1).to(device)
                
                fake = generator(z).detach()
                X = X.to(device)
                
                discriminator.trainable = True
                
                disc_loss = (-torch.mean(discriminator(X)) + torch.mean(discriminator(torch.cat([fake, Vc], dim=1)))).to(device)
                disc_loss = disc_loss + np.random.normal(0, std)

                fake_super = supervisor(torch.cat([fake, Vc], dim=1)).to(device)
                disc2_loss = (-torch.mean(discriminator2(X)) + torch.mean(discriminator2(torch.cat([fake_super, Vc], dim=1)))).to(device)
                disc2_loss = disc2_loss + np.random.normal(0, std)
                
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

                    gen_loss = (alpha_r * gen_loss1 + (1 - alpha_r) * supervisor_loss)
                    
                    supervisor_loss2 = ((-torch.mean(discriminator2(torch.cat([supervisor(torch.cat([generator(z), Vc], dim=1).detach()),\
                        Vc], dim=1)))) + lambda1 * loss(supervisor(torch.cat([generator(z), Vc], dim=1).detach()), X[:,:-dim_Vc])).to(device)
                    
                    
                    optimizer_G.zero_grad()
                    gen_loss.backward()
                    optimizer_G.step()
                    
                    optimizer_S.zero_grad()
                    supervisor_loss2.backward()
                    optimizer_S.step()


            epsilon_bar_array.append(eps)

            # scheduler_G.step()
            # scheduler_D.step()
            # scheduler_S.step()
            # scheduler_D2.step()


            epochs.set_description('Discriminator Loss: %.5f || Discriminator 2 Loss: %.5f || Generator Loss: %.5f || Supervisor Loss: %.5f' %\
                (disc_loss.item(), disc2_loss.item(), gen_loss.item(), supervisor_loss2.item()))
            loss_array.append([disc_loss.item(), disc2_loss.item(), gen_loss.item(), supervisor_loss2.item()])

        return generator, supervisor, loss_array, discriminator, discriminator2, epsilon_bar_array
    
    elif privacy_type == 'TDDP':
        alpha = 2
        delta = 0.1 #or 1/batch_size
        C = 1
        h = X_emb.shape[1]
        k = 4
        # gamma = (C * h + 1) * (b_d1**k * h**(k-1) + b_d1) + sum([(b_d1**i * h**(i-1)) for i in range(1, k-1)])
        Sk = sum([(b_d1**i * h**(i-1)) for i in range(1, k-1)])
        gamma = 0.5 * (h-1)/(h) * b_d1 * ((b_d1**k * (C*h + 1) * h**(k-1) + Sk + b_d1 * (C*h + 1))/
                        (b_d1**(k+1) * (C*h + 1)**2 * h**(k-1) +  b_d1 * (C*h + 1) * Sk)) + 2*b_d1
        sensitivity = 4 * gamma /batch_size
        n_iter = len(X_emb) // batch_size
       
        epsilon_bar_array = []

        for epoch in epochs:
            epsilon_d_array = []
            
            for batch_idx, X in enumerate(loader_g):
                loss = nn.MSELoss()
                loss_privacy_mse = nn.MSELoss()
                # loss_f_mse = nn.MSELoss()
                jsd = JSD()
                batch_size = X.size(0)

                Vc = X[:, -dim_Vc:].to(device)
                
                noise = torch.FloatTensor(dclProcess(batch_size - 1, dim_noise)).to(device)
                z = torch.cat([noise, Vc], dim=1).to(device)
                fake = generator(z).detach()
                X = X.to(device)
                
                r = np.random.choice(np.arange(batch_size))
                fake_comma = fake.clone()
                fake_comma[r, :] = torch.FloatTensor(z_func(fake_comma[r, :]))


                discriminator.trainable = True

                # epsilon_hat = torch.abs(privacyOperator(fake.to(device))).to(device)
                # epsilon_bar = torch.mean(epsilon_hat)
   
                # loss_privacy = (loss_privacy_mse(epsilon_bar, torch.FloatTensor([eps]).to(device)) + \
                #                 loss_f_mse(torch.sum(torch.sum(fake, 0) / torch.sum(fake_comma, 0)), torch.FloatTensor([h]))).to(device)

                
                
                # var_array = [np.sqrt((4 * q * alpha**2 * (sensitivity)**2 * np.sqrt(2 * n_iter * np.log(1/delta))) / (2 * (alpha-1) * j)) for j in epsilon_hat.detach().cpu().numpy()]
                # privacy_noise = torch.FloatTensor([np.random.normal(0, np.sqrt(sum(var_array)))])
                
                
                epsilon_tilde = privacyOperator(fake.to(device)).to(device)
                epsilon_bar = torch.mean(epsilon_tilde)
                # eps_hat = (np.maximum(np.repeat(epsilon_tilde.detach().cpu().numpy().reshape(-1, 1), 100, axis=1).T, eps) \
                #             + ornstein_uhlenbeck_process(batch_size, 100, eps)[:, 1:])/2
                # eps_hat = np.mean(eps_hat[:, -1])
                
                I, _ = compute_integral(100, len(epsilon_tilde), 1, epsilon_tilde.detach().cpu().numpy())
                eps_hat = np.mean(np.minimum(np.abs(I[:, -1]), eps))
                epsilon_d_array.append(eps_hat)
                
                
                std = np.sqrt((4 * q * alpha**2 * (sensitivity)**2 * np.sqrt(2 * n_iter * np.log(1/delta))) / (2 * (alpha-1) * eps_hat)) 
                privacy_noise = torch.FloatTensor([np.random.normal(0, std)])
                
                
                disc_loss = (-torch.mean(discriminator(X)) + torch.mean(discriminator(torch.cat([fake, Vc], dim=1))) + privacy_noise).to(device)

                fake_super = supervisor(torch.cat([fake, Vc], dim=1)).to(device)
                disc2_loss = (-torch.mean(discriminator2(X)) + torch.mean(discriminator2(torch.cat([fake_super, Vc], dim=1))) + privacy_noise).to(device) 

                optimizer_D.zero_grad()
                disc_loss.backward()
                optimizer_D.step()

                optimizer_D2.zero_grad()
                disc2_loss.backward()
                optimizer_D2.step()
                
                eps_alpha = 0.7
                loss_privacy = (eps_alpha * loss_privacy_mse(epsilon_bar, torch.FloatTensor([eps]).to(device)) +\
                                (1 - eps_alpha) * jsd(fake, X[:,:-dim_Vc])).to(device)
                    # (1-eps_alpha) * loss_f_mse(torch.abs(torch.mean(discriminator(torch.cat([fake, Vc], dim=1)).detach()) - torch.mean(discriminator(torch.cat([fake_comma, Vc], dim=1)).detach())),\
                                            # torch.abs(privacyOperator(fake.to(device))).to(device)[r])).to(device)

                optimizer_privacy.zero_grad()
                loss_privacy.backward()
                optimizer_privacy.step()
                
                
                for dp in discriminator.parameters():
                            dp.data.clamp_(-b_d1, b_d1)

                for dp in discriminator2.parameters():
                            dp.data.clamp_(-b_d2, b_d2)
                
                        
                if batch_idx % 2 == 0:
                    discriminator.trainable = False

                    gen_loss1 = -torch.mean(discriminator(torch.cat([generator(z), Vc], dim=1))).to(device)
                    supervisor_loss = (-torch.mean(discriminator2(torch.cat([supervisor(torch.cat([generator(z), Vc], dim=1).detach()), Vc], dim=1))) +\
                                        lambda1 * loss(supervisor(torch.cat([generator(z), Vc], dim=1).detach()), X[:,:-dim_Vc])).to(device)

                    gen_loss = (alpha_r * gen_loss1 + (1 - alpha_r) * supervisor_loss)
                    
                    optimizer_G.zero_grad()
                    gen_loss.backward()
                    optimizer_G.step()

                    
                    supervisor_loss2 = ((-torch.mean(discriminator2(torch.cat([supervisor(torch.cat([generator(z), Vc], dim=1).detach()),\
                        Vc], dim=1)))) + lambda1 * loss(supervisor(torch.cat([generator(z), Vc], dim=1).detach()), X[:,:-dim_Vc])).to(device)
                    
                    optimizer_S.zero_grad()
                    supervisor_loss2.backward()
                    optimizer_S.step()
                
                
                
            epsilon_bar_array.append(epsilon_bar.detach().cpu().numpy())

            # scheduler_G.step()
            # scheduler_D.step()
            # scheduler_S.step()
            # scheduler_D2.step()


            epochs.set_description('Discriminator Loss: %.5f || Discriminator 2 Loss: %.5f || Generator Loss: %.5f || Supervisor Loss: %.5f' %\
                (disc_loss.item(), disc2_loss.item(), gen_loss.item(), supervisor_loss2.item()))
            loss_array.append([disc_loss.item(), disc2_loss.item(), gen_loss.item(), supervisor_loss2.item()])
            
        print(f'TRGAN with ({np.mean(epsilon_d_array)}, {delta})-differential privacy')

        return generator, supervisor, loss_array, discriminator, discriminator2, epsilon_bar_array



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
        if M > 1:
            Z1[:, i] = (Z1[:, i] - np.mean(Z1[:, i])) / np.std(Z1[:, i])

        X[:,i+1] = X[:, i] - 1/theta * X[:,i] * dt + np.sqrt((1 - (X[:, i])**2)/(theta * (delta + 1))) * np.sqrt(dt) * Z1[:, i]
            
        if (X[:,i+1] > 1).any():
            X[np.where(X[:,i+1] >= 1)[0], i+1] = 0.9999

        if (X[:,i+1] < -1).any():
            X[np.where(X[:,i+1] <= -1)[0], i+1] = -0.9999 
            
        time[i+1] = time[i] + dt

    return X.T


def ornstein_uhlenbeck_process(N, M, eps):
    theta = 5
    sigma = 1
    mu = eps
    T = 1
    Z1 = np.random.normal(0.0, 1.0, [M, N])
    X = np.zeros([M, N + 1])
    
    dt = T / float(N)
    
    for t in range(N):
        if M > 1:
            Z1[:, t] = (Z1[:, t] - np.mean(Z1[:, t])) / np.std(Z1[:, t])
            
        X[:, t+1] = X[:, t] + theta*(mu - X[:, t]) * dt + sigma * np.sqrt(dt) * Z1[:, t]
    
    return X


def compute_integral(Npath, Nsteps, T, f):
    Z = np.random.normal(0, 1, [Npath, Nsteps])
    W = np.zeros([Npath, Nsteps+1])
    
    I = np.zeros([Npath, Nsteps+1])
    
    dt = T/Nsteps
    
    for i in range(Nsteps):
        if Npath > 1:
            Z[:, i] = (Z[:, i] - np.mean(Z[:, i])) / np.std(Z[:, i])
            
        W[:, i+1] = W[:, i] + np.sqrt(dt) * Z[:, i]
        I[:, i+1] = I[:, i] + f[i] * (W[:, i+1] - W[:, i])
        
    return I, W
    
    
class JSD(nn.Module):
    def __init__(self):
        super(JSD, self).__init__()
        self.kl = nn.KLDivLoss(reduction='batchmean', log_target=True)

    def forward(self, p: torch.tensor, q: torch.tensor):
        p, q = F.softmax(p.view(-1, p.size(-1))), F.softmax(q.view(-1, q.size(-1)))
        m = (0.5 * (p + q)).log()
        
        return 0.5 * (self.kl(m, p.log()) + self.kl(m, q.log()))