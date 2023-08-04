import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sys
import torch

from matplotlib import gridspec
from numba import njit, prange
from pyts.utils import segmentation
from pyts.preprocessing.discretizer import KBinsDiscretizer

import itertools
import tsia.plot
import tsia.markov
import tsia.network_graph

MTF_STRATEGY = 'quantile'
PATH_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
paths_dict = {
              'opportunity': os.path.join(PATH_DATA, 'opportunity'),
              'mhealthy': os.path.join(PATH_DATA, 'mhealthy'),
              'pamap2': os.path.join(PATH_DATA, 'pamap2'),
              'uci-har': os.path.join(PATH_DATA, 'uci-har'),
              }

def markov_discretize(timeseries, n_bins, strategy=MTF_STRATEGY):
    discretizer = KBinsDiscretizer(n_bins=n_bins, strategy=strategy)
    X = timeseries.reshape(1, -1)
    X_binned = discretizer.fit_transform(X)[0]
    bin_edges = discretizer._compute_bins(X, 1, n_bins, strategy)
    bin_edges = np.hstack((timeseries.min(),
                           bin_edges[0],
                           timeseries.max()))

    return X_binned, bin_edges

def markov_transition_matrix_dimension(all_binned,dim0_binned,dim1_binned):
    n_bins = np.max(all_binned) + 1
    X_mtm = np.zeros((n_bins, n_bins))
    n_timestamps = dim0_binned.shape[0]
    for j in prange(n_timestamps - 1):
        X_mtm[dim0_binned[j], dim1_binned[j]] += 1
    return X_mtm

def markov_transition_field(dim0_binned,dim1_binned, dim_mtm, n_timestamps, n_bins):
    dim_mtf = np.zeros((n_timestamps, n_timestamps))
    for i in prange(n_timestamps):
        for j in prange(n_timestamps):
            dim_mtf[i, j] = dim_mtm[dim0_binned[i], dim1_binned[j]]

    return dim_mtf

def _aggregated_markov_transition_field(X_mtf, image_size,
                                        start, end):
    X_amtf = np.empty((image_size, image_size))
    for j in prange(image_size):
        for k in prange(image_size):
            X_amtf[j, k] = np.mean(
                 X_mtf[start[j]:end[j], start[k]:end[k]]
            )
    return X_amtf

def average_markov_transition_field(dim_mtf,image_size,n_timestamps,overlapping=False):
    window_size, remainder = divmod(n_timestamps, image_size)
    if remainder == 0:
        dim_amtf = np.reshape(
        dim_mtf, (image_size, window_size, image_size, window_size)
        ).mean(axis=(1, 3))
    else:
        window_size += 1
        start, end, _ = segmentation(
            n_timestamps, window_size, overlapping, image_size
        )
        dim_amtf = _aggregated_markov_transition_field(
             dim_mtf, image_size, start, end
        )
    return dim_amtf

def dimension_markov_transition_field(all_binned,dim0_binned,dim1_binned, n_timestamps, n_bins, image_size):

    dim_mtm = markov_transition_matrix_dimension(all_binned,dim0_binned,dim1_binned)
    dim_mtm = tsia.markov.markov_transition_probabilities(dim_mtm)
    dim_mtf = markov_transition_field(dim0_binned,dim1_binned,dim_mtm, n_timestamps,n_bins)
    dim_amtf = average_markov_transition_field(dim_mtf,image_size=image_size,n_timestamps=n_timestamps)

    return dim_amtf

def bulid_markov_transition(data,image_size,n_bins,dataname,filename):
    n_samples,seq_length,n_dimension = data.shape
    ans = []
    for i in range(0, n_samples):
        temp = []
        all_binned, bin_edges_all = markov_discretize(data[i,:,:],n_bins)
        binned_data = all_binned.reshape(seq_length,n_dimension)
        for j in itertools.permutations(range(n_dimension),2):
            tag_dim0 = binned_data[:,j[0]]
            tag_dim1 = binned_data[:,j[1]]
            dim_mak = dimension_markov_transition_field(all_binned, tag_dim0,tag_dim1,n_timestamps=seq_length,n_bins=n_bins,image_size=image_size)
            temp.append(dim_mak)
        ans.append(temp)

    ans = np.mean(ans,axis=1)
    ans = torch.Tensor(ans)
    ans = ans[:,np.newaxis,:,:]
    save_image(ans,dataname,filename,n_bins,image_size)


def preprocess_data(train_data, preprocess=None):
    if preprocess == 'standardize':
        return __standardize(train_data)
    elif preprocess == 'scale':
        return __scale(train_data)
    else:
        return train_data

def __standardize(train_data):
    print('standardizing data...')
    data_mean = np.mean(train_data)
    data_std = np.std(train_data)
    return (train_data - data_mean) / data_std

def __scale(train_data):
    print('scaling data...')
    for i in range(train_data.shape[2]):
        min_tri = np.min(train_data[:, :, i])
        train_data[:, :, i] = train_data[:, :, i] - min_tri
        max_tri = np.max(train_data[:, :, i])
        train_data[:, :, i] = train_data[:, :, i] / max_tri

    return train_data

def save_image(imag_data, dataset_name, filename, n_bins, image_size):
    file_image = '{}{}image_{}_{}_{}_train.npy'.format(paths_dict[dataset_name], os.sep, filename, n_bins, image_size)
    np.save(file_image, imag_data)

def load_saved_image(dataset_name,filename,n_bins,image_size):
    file_image = '{}{}image_{}_{}_{}_train.npy'.format(paths_dict[dataset_name], os.sep, filename, n_bins, image_size)
    if os.path.isfile(file_image):
        print('Loading image-data from '+ paths_dict[dataset_name])
        data_image = np.load(file_image)
        data_image = preprocess_data(data_image,preprocess='standardize')
        return data_image
    return None





