import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import copy

import torch
from torch import optim
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.special import rel_entr, kl_div
from scipy.spatial.distance import jensenshannon
from scipy.stats import kstest, ks_2samp, wasserstein_distance
import scipy.stats as sts
from tqdm import tqdm

from TRGAN.TRGAN_main_V2 import *
from TRGAN.encoders import *
import TRGAN.TRGAN_train_load_modules as trgan_train
from TRGAN.evaluation_metrics import *


from sdv.single_table import CopulaGANSynthesizer
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer
from sdv.single_table import TVAESynthesizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def plot_hist_categorical(data, synth_df, feat_name, tick=50):
    plt.subplots(figsize=(10, 5), dpi=100)

    plt.hist(data[feat_name], alpha=0.6, label='Real', bins=np.arange(0, data[feat_name].value_counts().shape[0], tick), color='black', rwidth=0.8, density=True)
    plt.hist(synth_df[feat_name], alpha=0.6, label='Synth', bins=np.arange(0, data[feat_name].value_counts().shape[0], tick), color='red', rwidth=0.8, density=True)

    plt.legend()
    plt.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    plt.xlabel(f'Bins of categorical feature "{feat_name}"')
    plt.ylabel('Counts')

    plt.show()
    
    
def plot_hist_numerical(data, synth_df, feat_name, tick=60):
    plt.subplots(figsize=(10, 5), dpi=100)
    bins_range = np.arange(np.min(data[feat_name]), np.max(data[feat_name]), round(np.max(data[feat_name]) - np.min(data[feat_name])) / tick)
    
    plt.hist(data[feat_name], bins=bins_range, label='Real', alpha=0.6, density=True, color='black')
    plt.hist(synth_df[feat_name], bins=bins_range, label='Synth', alpha=0.6, density=True, color='red')

    plt.legend()
    plt.xlabel(f'"{feat_name}" values')
    plt.ylabel('Density')
    plt.show()


def plot_hist_onehot(data: pd.DataFrame, synth_df: pd.DataFrame, feat_name: str):
    plt.subplots(figsize=(10, 5), dpi=100)

    if data[feat_name].dtype == 'float' or data[feat_name].dtype == 'int':
        plt.bar(data[feat_name].value_counts().index.values.astype(int).astype(str), data[feat_name].value_counts().values/(np.sum(data[feat_name].value_counts().values)),\
                                                            color='black', alpha=0.6, label='Real')
        plt.bar(synth_df[feat_name].value_counts().index.values.astype(int).astype(str), synth_df[feat_name].value_counts().values/(np.sum(synth_df[feat_name].value_counts().values)),\
                                                            color='red', alpha=0.6, label='TRGAN')

        plt.xticks(data[feat_name].value_counts().index.values.astype(str)[::], labels=data[feat_name].value_counts().index.values.astype(str)[::], rotation=45)
        # plt.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
        plt.legend()
        plt.ylabel('Density')
        plt.xlabel(f'"{feat_name}" values')
        plt.show()
        
    elif data[feat_name].dtype == 'object':
        plt.bar(data[feat_name].value_counts().index.values.astype(str), data[feat_name].value_counts().values/(np.sum(data[feat_name].value_counts().values)),\
                                                            color='black', alpha=0.6, label='Real')
        plt.bar(synth_df[feat_name].value_counts().index.values.astype(str), synth_df[feat_name].value_counts().values/(np.sum(synth_df[feat_name].value_counts().values)),\
                                                            color='red', alpha=0.6, label='TRGAN')

        plt.xticks(data[feat_name].value_counts().index.values.astype(str)[::], labels=data[feat_name].value_counts().index.values.astype(str)[::], rotation=45)
        # plt.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
        plt.legend()
        plt.ylabel('Density')
        plt.xlabel(f'"{feat_name}" values')
        plt.show()
        
    else:
        print('Incorrect data type. Available "float, int and object".')


'''
Copula GAN
'''

def copulagan(data: pd.DataFrame, metadata, cat_columns=['mcc'], epochs=10, n_samples=10_000, load=False):
    
    if load:
        synthesizer = CopulaGANSynthesizer(metadata, epochs=epochs)
        synthesizer = synthesizer.load('CopulaGAN.pkl')
    else:
        synthesizer = CopulaGANSynthesizer(metadata, epochs=epochs)
        synthesizer.fit(data)
        synthesizer.save('CopulaGAN.pkl')

    synth_copulagan = synthesizer.sample(num_rows=n_samples)

    synth_df_cat_copulagan = pd.get_dummies(synth_copulagan[cat_columns], columns=cat_columns)

    return synth_copulagan, synth_df_cat_copulagan


'''
CTGAN
'''

def ctgan(data: pd.DataFrame, metadata, cat_columns=['mcc'], epochs=10, n_samples=10_000, load=False):

    if load:
        synthesizer = CTGANSynthesizer(metadata, epochs=epochs)
        synthesizer = synthesizer.load('CTGAN.pkl')
    else:
        synthesizer = CTGANSynthesizer(metadata, epochs=epochs)
        synthesizer.fit(data)
        synthesizer.save('CTGAN.pkl')
 
    synth_ctgan = synthesizer.sample(num_rows=n_samples)

    synth_df_cat_ctgan = pd.get_dummies(synth_ctgan[cat_columns], columns=cat_columns)

    return synth_ctgan, synth_df_cat_ctgan


'''
TVAE
'''

def tvae(data: pd.DataFrame, metadata, cat_columns=['mcc'], epochs=10, n_samples=10_000, load=False):

    if load:
        synthesizer = TVAESynthesizer(metadata, epochs=epochs)
        synthesizer = synthesizer.load('TVAE.pkl')
    else:
        synthesizer = TVAESynthesizer(metadata, epochs=epochs)
        synthesizer.fit(data)
        synthesizer.save('TVAE.pkl')
    

    synth_tvae = synthesizer.sample(num_rows=n_samples)

    synth_df_cat_tvae= pd.get_dummies(synth_tvae[cat_columns], columns=cat_columns)

    return synth_tvae, synth_df_cat_tvae


def compare_numerical(data, synth_df, metadata, cat_columns, epochs=10, n_samples=10_000, comp_col='amount'):
    synth_copulagan, synth_copulagan_cat = copulagan(data, metadata, cat_columns, epochs, n_samples)
    synth_ctgan, synth_ctgan_cat = ctgan(data, metadata, cat_columns, epochs, n_samples)
    synth_tvae, synth_tvae_cat = tvae(data, metadata, cat_columns, epochs, n_samples)

    fig, axs = plt.subplots(2, 2, figsize=(20, 10), dpi=100)

    axs[0, 0].hist(data['amount'], bins=30, label='Real', alpha=0.7, density=True, color='black')
    axs[0, 0].hist(synth_df['amount'], bins=30, label='TRGAN', alpha=0.7, density=True, color='red')
    axs[0, 0].legend()
    axs[0, 0].set_xlabel('Amount')
    axs[0, 0].set_ylabel('Frequencies')
    axs[0, 0].set_title('TRGAN')

    axs[0, 1].hist(data['amount'], bins=30, label='Real', alpha=0.7, density=True, color='black')
    axs[0, 1].hist(synth_ctgan['amount'], bins=30, label='CTGAN', alpha=0.7, density=True, color='lightblue')
    axs[0, 1].legend()
    axs[0, 1].set_xlabel('Amount')
    axs[0, 1].set_ylabel('Frequencies')
    axs[0, 1].set_title('CTGAN')

    axs[1, 0].hist(data['amount'], bins=30, label='Real', alpha=0.7, density=True, color='black')
    axs[1, 0].hist(synth_copulagan['amount'], bins=30, label='CopulaGAN', alpha=0.7, density=True, color='orange')
    axs[1, 0].legend()
    axs[1, 0].set_xlabel('Amount')
    axs[1, 0].set_ylabel('Frequencies')
    axs[1, 0].set_title('CopulaGAN')

    axs[1, 1].hist(data['amount'], bins=30, label='Real', alpha=0.7, density=True, color='black')
    axs[1, 1].hist(synth_tvae['amount'], bins=30, label='TVAE', alpha=0.7, density=True, color='green')
    axs[1, 1].legend()
    axs[1, 1].set_xlabel('Amount')
    axs[1, 1].set_ylabel('Frequencies')
    axs[1, 1].set_title('TVAE')

    plt.subplots_adjust(hspace=0.3)
    plt.show()

    eval_num = evaluate_numerical([data[comp_col], synth_df[comp_col], synth_ctgan[comp_col], synth_copulagan[comp_col], synth_tvae[comp_col]],\
                 ['Real', 'TRGAN', 'CTGAN', 'CopulaGAN', 'TVAE'])
    
    return eval_num

def compare_numerical_w_banksformer(data, synth_df, metadata, cat_columns, synth_banks, epochs=10, n_samples=10_000, comp_col='amount'):
    synth_copulagan, synth_copulagan_cat = copulagan(data, metadata, cat_columns, epochs, n_samples)
    synth_ctgan, synth_ctgan_cat = ctgan(data, metadata, cat_columns, epochs, n_samples)
    # synth_tvae, synth_tvae_cat = tvae(data, metadata, cat_columns, epochs, n_samples)
    synth_banksformer = synth_banks

    fig, axs = plt.subplots(2, 2, figsize=(20, 10), dpi=100)

    axs[0, 0].hist(data['amount'], bins=30, label='Real', alpha=0.7, density=True, color='black')
    axs[0, 0].hist(synth_df['amount'], bins=30, label='TRGAN', alpha=0.7, density=True, color='red')
    axs[0, 0].legend()
    axs[0, 0].set_xlabel('Amount')
    axs[0, 0].set_ylabel('Density')
    axs[0, 0].set_title('TRGAN')

    axs[0, 1].hist(data['amount'], bins=30, label='Real', alpha=0.7, density=True, color='black')
    axs[0, 1].hist(synth_ctgan['amount'], bins=30, label='CTGAN', alpha=0.7, density=True, color='lightblue')
    axs[0, 1].legend()
    axs[0, 1].set_xlabel('Amount')
    axs[0, 1].set_ylabel('Density')
    axs[0, 1].set_title('CTGAN')

    axs[1, 0].hist(data['amount'], bins=30, label='Real', alpha=0.7, density=True, color='black')
    axs[1, 0].hist(synth_copulagan['amount'], bins=30, label='CopulaGAN', alpha=0.7, density=True, color='orange')
    axs[1, 0].legend()
    axs[1, 0].set_xlabel('Amount')
    axs[1, 0].set_ylabel('Density')
    axs[1, 0].set_title('CopulaGAN')

    axs[1, 1].hist(data['amount'], bins=30, label='Real', alpha=0.7, density=True, color='black')
    axs[1, 1].hist(synth_banksformer['amount'], bins=30, label='Banksformer', alpha=0.7, density=True, color='green')
    axs[1, 1].legend()
    axs[1, 1].set_xlabel('Amount')
    axs[1, 1].set_ylabel('Density')
    axs[1, 1].set_title('Banksformer')

    plt.subplots_adjust(hspace=0.3)
    plt.show()

    eval_num = evaluate_numerical([data[comp_col], synth_df[comp_col], synth_ctgan[comp_col], synth_copulagan[comp_col], synth_banksformer[comp_col]],\
                 ['Real', 'TRGAN', 'CTGAN', 'CopulaGAN', 'Banksformer'])
    
    return eval_num

def compare_categorical(data, synth_df, synth_df_cat, X_oh, metadata, cat_columns, epochs=10, n_samples=10_000, comp_col='mcc', contig_cols=['mcc', 'customer']):
    synth_copulagan, synth_copulagan_cat = copulagan(data, metadata, cat_columns, epochs, n_samples)
    synth_ctgan, synth_ctgan_cat = ctgan(data, metadata, cat_columns, epochs, n_samples)
    synth_tvae, synth_tvae_cat = tvae(data, metadata, cat_columns, epochs, n_samples)

    fig, axs = plt.subplots(2, 2, figsize=(20, 10), dpi=100)

    # axs[0, 0].hist(data['mcc'], bins=30, label='Real', alpha=0.7, density=True, color='black')
    # axs[0, 0].hist(synth_df['mcc'], bins=30, label='Synth TRGAN', alpha=0.7, density=True, color='red')
    axs[0, 0].bar(np.sort(data['mcc'].value_counts().index.values).astype(str), data['mcc'].value_counts().sort_index().values, color='black', alpha=0.6, label='Real')
    axs[0, 0].bar(np.sort(synth_df['mcc'].value_counts().index.values).astype(str), synth_df['mcc'].value_counts().sort_index().values, color='red', alpha=0.6, label='TRGAN')
    axs[0, 0].set_xticks(data['mcc'].value_counts().sort_index().index.values.astype(str)[::2])
    axs[0, 0].set_xticklabels(data['mcc'].value_counts().sort_index().index.values.astype(str)[::2], rotation=45)

    axs[0, 0].legend()
    axs[0, 0].set_xlabel('MCC')
    axs[0, 0].set_ylabel('Frequencies')
    axs[0, 0].set_title('TRGAN')

    # axs[0, 1].hist(data['mcc'], bins=30, label='Real', alpha=0.7, density=True, color='black')
    # axs[0, 1].hist(synth_ctgan['mcc'], bins=30, label='CTGAN', alpha=0.7, density=True, color='lightblue')
    axs[0, 1].bar(np.sort(data['mcc'].value_counts().index.values).astype(str), data['mcc'].value_counts().sort_index().values, color='black', alpha=0.6, label='Real')
    axs[0, 1].bar(np.sort(synth_ctgan['mcc'].value_counts().index.values).astype(str), synth_ctgan['mcc'].value_counts().sort_index().values, color='lightblue', alpha=0.6, label='CTGAN')

    axs[0, 1].set_xticks(data['mcc'].value_counts().sort_index().index.values.astype(str)[::2])
    axs[0, 1].set_xticklabels(data['mcc'].value_counts().sort_index().index.values.astype(str)[::2], rotation=45)
    axs[0, 1].legend()
    axs[0, 1].set_xlabel('MCC')
    axs[0, 1].set_ylabel('Frequencies')
    axs[0, 1].set_title('CTGAN')

    # axs[1, 0].hist(data['mcc'], bins=30, label='Real', alpha=0.7, density=True, color='black')
    # axs[1, 0].hist(synth_copulagan['mcc'], bins=30, label='CopulaGAN', alpha=0.7, density=True, color='orange')
    axs[1, 0].bar(np.sort(data['mcc'].value_counts().index.values).astype(str), data['mcc'].value_counts().sort_index().values, color='black', alpha=0.6, label='Real')
    axs[1, 0].bar(np.sort(synth_copulagan['mcc'].value_counts().index.values).astype(str), synth_copulagan['mcc'].value_counts().sort_index().values, color='orange', alpha=0.6, label='CopulaGAN')

    axs[1, 0].set_xticks(data['mcc'].value_counts().sort_index().index.values.astype(str)[::2])
    axs[1, 0].set_xticklabels(data['mcc'].value_counts().sort_index().index.values.astype(str)[::2], rotation=45)
    axs[1, 0].legend()
    axs[1, 0].set_xlabel('MCC')
    axs[1, 0].set_ylabel('Frequencies')
    axs[1, 0].set_title('CopulaGAN')

    # axs[1, 1].hist(data['mcc'], bins=30, label='Real', alpha=0.7, density=True, color='black')
    # axs[1, 1].hist(synth_tvae['mcc'], bins=30, label='TVAE', alpha=0.7, density=True, color='green')
    axs[1, 1].bar(np.sort(data['mcc'].value_counts().index.values).astype(str), data['mcc'].value_counts().sort_index().values, color='black', alpha=0.6, label='Real')
    axs[1, 1].bar(np.sort(synth_tvae['mcc'].value_counts().index.values).astype(str), synth_tvae['mcc'].value_counts().sort_index().values, color='green', alpha=0.6, label='TVAE')
    
    axs[1, 1].set_xticks(data['mcc'].value_counts().sort_index().index.values.astype(str)[::2])
    axs[1, 1].set_xticklabels(data['mcc'].value_counts().sort_index().index.values.astype(str)[::2], rotation=45)
    axs[1, 1].legend()
    axs[1, 1].set_xlabel('MCC')
    axs[1, 1].set_ylabel('Frequencies')
    axs[1, 1].set_title('TVAE')

    plt.subplots_adjust(hspace=0.3)
    plt.show()



    # pca1 = PCA(n_components=2)
    # pca2 = PCA(n_components=2)

    # data_transformed_pca = pca1.fit_transform(X_oh.iloc[:, :].values)
    # synth_pca = pca1.transform(synth_df_cat.iloc[:, :].values)
    # synth_pca_ctgan = pca1.transform(synth_ctgan_cat.values)
    # synth_pca_copulagan = pca1.transform(synth_copulagan_cat.values)
    # synth_pca_tvae = pca2.fit_transform(synth_tvae_cat.values)

    # idx_random = np.random.randint(0, len(data), 5000)

    # tsne1 = TSNE(n_components=2, perplexity = 80)
    # tsne2 = TSNE(n_components=2, perplexity = 80)
    # tsne3 = TSNE(n_components=2, perplexity = 80)
    # tsne4 = TSNE(n_components=2, perplexity = 80)
    # tsne5 = TSNE(n_components=2, perplexity = 80)

    # data_transformed_tsne = tsne1.fit_transform(X_oh.iloc[:, :13].values[idx_random])
    # synth_tsne = tsne2.fit_transform(synth_df_cat.iloc[:, :13].values[idx_random])
    # synth_tsne_ctgan = tsne3.fit_transform(synth_ctgan_cat.values[idx_random])
    # synth_tsne_copulagan = tsne4.fit_transform(synth_copulagan_cat.values[idx_random])
    # synth_tsne_tvae = tsne5.fit_transform(synth_tvae_cat.values[idx_random])

    # figure, axs = plt.subplots(1, 2, figsize=(15, 5), dpi=100)

    # axs[0].scatter(data_transformed_pca.T[0], data_transformed_pca.T[1], label='Real', alpha=0.4, s=20, color='black')
    # axs[0].scatter(synth_pca.T[0], synth_pca.T[1], label='Synth TRGAN', alpha=0.4, s=10, color='red')
    # axs[0].scatter(synth_pca_ctgan.T[0], synth_pca_ctgan.T[1], label='Synth CTGAN', alpha=0.4, s=10, color='green')
    # axs[0].scatter(synth_pca_copulagan.T[0], synth_pca_copulagan.T[1], label='Synth CopulaGAN', alpha=0.4, s=10, color='orange')
    # # axs[0].scatter(synth_pca_tvae.T[0], synth_pca_tvae.T[1], label='Synth TVAE', alpha=0.4, s=10, color='blue')

    # axs[0].legend()
    # axs[0].set_xlabel('$X_1$')
    # axs[0].set_ylabel('$X_2$')
    # axs[0].set_title('PCA')


    # axs[1].scatter(data_transformed_tsne.T[0], data_transformed_tsne.T[1], label='Real', s=20, alpha=1, color='black')
    # axs[1].scatter(synth_tsne.T[0], synth_tsne.T[1], label='Synth TRGAN', s=20, alpha=1, color='red')
    # axs[1].scatter(synth_tsne_ctgan.T[0], synth_tsne_ctgan.T[1], label='Synth CTGAN', s=20, alpha=1, color='green')
    # axs[1].scatter(synth_tsne_copulagan.T[0], synth_tsne_copulagan.T[1], label='Synth CopulaGAN', s=20, alpha=1, color='orange')
    # # axs[1].scatter(synth_tsne_tvae.T[0], synth_tsne_tvae.T[1], label='Synth TVAE', s=20, alpha=1, color='blue')

    # axs[1].legend()
    # axs[1].set_xlabel('$X_1$')
    # axs[1].set_ylabel('$X_2$')
    # axs[1].set_title('t-SNE')

    # plt.show()

    eval_cat = evaluate_categorical([data[comp_col], synth_df[comp_col], synth_ctgan[comp_col], synth_copulagan[comp_col], synth_tvae[comp_col]],\
                                     index=['Real', 'TRGAN', 'CTGAN', 'CopulaGAN', 'TVAE'],\
    data_cont_array=[data[contig_cols], synth_df[contig_cols], synth_ctgan[contig_cols], synth_copulagan[contig_cols], synth_tvae[contig_cols]])

    return eval_cat



def compare_categorical_w_banksformer(data, synth_df, synth_df_cat, X_oh, metadata, cat_columns, synth_banks,\
                                    epochs=2, n_samples=10_000, comp_col='mcc', contig_cols=['mcc', 'customer']):
    synth_copulagan, synth_copulagan_cat = copulagan(data, metadata, cat_columns, epochs, n_samples)
    synth_ctgan, synth_ctgan_cat = ctgan(data, metadata, cat_columns, epochs, n_samples)
    # synth_tvae, synth_tvae_cat = tvae(data, metadata, cat_columns, epochs, n_samples)
    synth_banksformer = synth_banks

    fig, axs = plt.subplots(2, 2, figsize=(20, 10), dpi=100)

    axs[0, 0].bar(np.sort(data['mcc'].value_counts().index.values).astype(str),\
            data['mcc'].value_counts().values/np.sum(data['mcc'].value_counts().values), color='black', alpha=0.6, label='Real')
    axs[0, 0].bar(np.sort(synth_df['mcc'].value_counts().index.values).astype(str),\
            synth_df['mcc'].value_counts().values/np.sum(synth_df['mcc'].value_counts().values), color='red', alpha=0.6, label='TRGAN')
    
    # axs[0, 0].set_xticks(data['mcc'].value_counts().index.values.astype(str)[::2])
    # axs[0, 0].set_xticklabels(data['mcc'].value_counts().index.values.astype(str)[::2], rotation=45)
    axs[0, 0].tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    axs[0, 0].legend()
    axs[0, 0].set_xlabel('MCC')
    axs[0, 0].set_ylabel('Density')
#     axs[0, 0].set_title('TRGAN')


    axs[0, 1].bar(np.sort(data['mcc'].value_counts().index.values).astype(str),\
            data['mcc'].value_counts().values/np.sum(data['mcc'].value_counts().values), color='black', alpha=0.6, label='Real')
    axs[0, 1].bar(np.sort(synth_ctgan['mcc'].value_counts().index.values).astype(str),\
            synth_ctgan['mcc'].value_counts().values/np.sum(synth_ctgan['mcc'].value_counts().values), color='lightblue', alpha=0.6, label='CTGAN')

    # axs[0, 1].set_xticks(data['mcc'].value_counts().index.values.astype(str)[::2])
    # axs[0, 1].set_xticklabels(data['mcc'].value_counts().index.values.astype(str)[::2], rotation=45)
    axs[0, 1].tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    axs[0, 1].legend()
    axs[0, 1].set_xlabel('MCC')
    axs[0, 1].set_ylabel('Density')
#     axs[0, 1].set_title('CTGAN')


    axs[1, 0].bar(np.sort(data['mcc'].value_counts().index.values).astype(str),\
            data['mcc'].value_counts().values/np.sum(data['mcc'].value_counts().values), color='black', alpha=0.6, label='Real')
    axs[1, 0].bar(np.sort(synth_copulagan['mcc'].value_counts().index.values).astype(str),\
            synth_copulagan['mcc'].value_counts().values/np.sum(synth_copulagan['mcc'].value_counts().values), color='orange', alpha=0.6, label='CopulaGAN')

    # axs[1, 0].set_xticks(data['mcc'].value_counts().index.values.astype(str)[::2])
    # axs[1, 0].set_xticklabels(data['mcc'].value_counts().index.values.astype(str)[::2], rotation=45)
    axs[1, 0].tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    axs[1, 0].legend()
    axs[1, 0].set_xlabel('MCC')
    axs[1, 0].set_ylabel('Density')
#     axs[1, 0].set_title('CopulaGAN')

    
    axs[1, 1].bar(np.sort(data['mcc'].value_counts().index.values).astype(str),\
            data['mcc'].value_counts().values/np.sum(data['mcc'].value_counts().values), color='black', alpha=0.6, label='Real')
    axs[1, 1].bar(np.sort(synth_banksformer['mcc'].value_counts().index.values).astype(str),\
            synth_banksformer['mcc'].value_counts().values/np.sum(synth_banksformer['mcc'].value_counts().values), color='green', alpha=0.6, label='Banksformer')
    
    # axs[1, 1].set_xticks(data['mcc'].value_counts().index.values.astype(str)[::2])
    # axs[1, 1].set_xticklabels(data['mcc'].value_counts().index.values.astype(str)[::2], rotation=45)
    axs[1, 1].tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    axs[1, 1].legend()
    axs[1, 1].set_xlabel('MCC')
    axs[1, 1].set_ylabel('Density')
#     axs[1, 1].set_title('Banksformer')

    plt.subplots_adjust(hspace=0.3)
    plt.show()

    eval_cat = evaluate_categorical([data[comp_col], synth_df[comp_col], synth_ctgan[comp_col], synth_copulagan[comp_col], synth_banksformer[comp_col]],\
                                     index=['Real', 'TRGAN', 'CTGAN', 'CopulaGAN', 'Banksformer'],\
    data_cont_array=[data[contig_cols], synth_df[contig_cols], synth_ctgan[contig_cols], synth_copulagan[contig_cols], synth_banksformer[contig_cols]])

    return eval_cat


def window_split(cl):
    cl_mcc_cplit = []
    for i in range(len(cl)-2):
        cl_mcc_cplit.append(cl.iloc[i:i+3].values)
    return cl_mcc_cplit

def join_in_one_array(array):
    res = []
    for i in array:
        for j in i:
            res.append(j)

    return res

def calculate_3_grams(data):
    split_data = data.groupby('customer')['mcc'].apply(lambda x: window_split(x))
    split_data = split_data.values
    split_data = join_in_one_array(split_data)

    split_data_df = pd.DataFrame(np.array(split_data), columns=['1', '2', '3'])
    split_data_df['ones'] = 1
    split_counts = split_data_df.groupby(['1', '2', '3']).count().sort_values(by='ones', ascending=False).iloc[:25].values.flatten()

    i_data = split_data_df.groupby(['1', '2', '3']).count().sort_values(by='ones', ascending=False).iloc[:25].index.values

    return i_data, split_counts


def calculate_3_grams_synth(data, idx_real):
    split_data = data.groupby('customer')['mcc'].apply(lambda x: window_split(x))
    split_data = split_data.values
    split_data = join_in_one_array(split_data)

    split_data_df = pd.DataFrame(np.array(split_data), columns=['1', '2', '3'])
    split_data_df['ones'] = 1

    idx_synth = split_data_df.groupby(['1', '2', '3']).count().sort_values(by='ones', ascending=False).iloc[:25].index
    
    split_counts = split_data_df.groupby(['1', '2', '3']).count().sort_values(by='ones', ascending=False).iloc[:25].loc[np.intersect1d(idx_synth,\
         idx_real)].sort_values(by='ones', ascending=False).values.flatten()

    i_data = split_data_df.groupby(['1', '2', '3']).count().sort_values(by='ones', ascending=False).iloc[:25].loc[np.intersect1d(idx_synth,\
         idx_real)].sort_values(by='ones', ascending=False).index.values

    return i_data, split_counts    

def calculate_idx_3_grams(i_data, all_comb):
    idx_real_pos = []

    for j in i_data:
        for i in range(len(all_comb)):
            if j == all_comb[i]:
                idx_real_pos.append(i)
                break
        else:
            continue

    return idx_real_pos


def compare_categorical_w_banksformer_3grams(data, synth_df, synth_banks):

    synth_banksformer = synth_banks


    real_3grams_idx, real_3grams_count = calculate_3_grams(data.sort_values(by=['customer'])[['mcc', 'customer']])
    trgan_3grams_idx, trgan_3grams_count = calculate_3_grams_synth(synth_df.sort_values(by=['customer'])[['mcc', 'customer']], real_3grams_idx)
    banks_3grams_idx, banks_3grams_count = calculate_3_grams_synth(synth_banksformer.sort_values(by=['customer'])[['mcc', 'customer']], real_3grams_idx)

    all_comb = real_3grams_idx

    #  np.sort(np.unique(np.hstack([real_3grams_idx, trgan_3grams_idx, ctgan_3grams_idx, copulagan_3grams_idx, banks_3grams_idx])))
    
    real_3grams_idx = calculate_idx_3_grams(real_3grams_idx, all_comb)
    trgan_3grams_idx = calculate_idx_3_grams(trgan_3grams_idx, all_comb)
    banks_3grams_idx = calculate_idx_3_grams(banks_3grams_idx, all_comb)

    real_3grams_count = real_3grams_count/np.sum(real_3grams_count)
    trgan_3grams_count = trgan_3grams_count/np.sum(trgan_3grams_count)
    banks_3grams_count = banks_3grams_count/np.sum(banks_3grams_count)


    fig, axs = plt.subplots(1, 2, figsize=(20, 5), dpi=100)

    axs[0].bar(real_3grams_idx, real_3grams_count, color='black', alpha=0.6, label='Real')
    axs[0].bar(trgan_3grams_idx, trgan_3grams_count, color='red', alpha=0.6, label='TRGAN')
    
    axs[0].tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    axs[0].legend(fontsize=12)
    axs[0].set_xlabel('3-grams', fontsize=16)
    axs[0].set_ylabel('Density', fontsize=16)
    axs[0].tick_params(labelsize=13)
    # axs[0].set_ylim((0, 0.17))
#     axs[0, 0].set_title('TRGAN')
    
    axs[1].bar(real_3grams_idx, real_3grams_count, color='black', alpha=0.6, label='Real')
    axs[1].bar(banks_3grams_idx, banks_3grams_count, color='green', alpha=0.6, label='Banksformer')

    axs[1].tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    axs[1].legend(fontsize=12)
    axs[1].set_xlabel('3-grams', fontsize=16)
    axs[1].set_ylabel('Density', fontsize=16)
    axs[1].tick_params(labelsize=13)
    # axs[1].set_ylim((0, 0.17))
#     axs[1, 1].set_title('Banksformer')

    plt.subplots_adjust(hspace=0.15, wspace=0.16)
    # plt.savefig('synth_3grams_czech.pdf', dpi=300)
    plt.show()


    z1 = list(zip(trgan_3grams_idx, trgan_3grams_count))
    trgan_3grams_count2 = np.array(sorted(np.concatenate([z1, list(zip(np.setdiff1d(real_3grams_idx, trgan_3grams_idx),\
                                                np.zeros(len(np.setdiff1d(real_3grams_idx, trgan_3grams_idx)))))]), key=lambda x: x[0])).T[1]
    
    z2 = list(zip(banks_3grams_idx, banks_3grams_count))
    banks_3grams_count2 = np.array(sorted(np.concatenate([z2, list(zip(np.setdiff1d(real_3grams_idx, banks_3grams_idx),\
                                                np.zeros(len(np.setdiff1d(real_3grams_idx, banks_3grams_idx)))))]), key=lambda x: x[0])).T[1]

    display(pd.DataFrame(np.array([[jensenshannon(real_3grams_count, real_3grams_count), jensenshannon(real_3grams_count, trgan_3grams_count2),\
                                    jensenshannon(real_3grams_count, banks_3grams_count2)],\
                                   [wasserstein_distance(real_3grams_count, real_3grams_count), wasserstein_distance(real_3grams_count, trgan_3grams_count),\
                                    wasserstein_distance(real_3grams_count, banks_3grams_count)]]).T,\
                        columns=['D_JS', 'W_1'], index=['Real', 'TRGAN', 'Banksformer']))

