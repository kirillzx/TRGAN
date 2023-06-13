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