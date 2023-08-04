import os
import numpy as np
import pandas as pd
from utils import dataimport as di
from utils import markov_transition as mk

build_dict_la = {'opportunity': di.build_opportunity_labeled,'mhealthy':di.build_mhealth_labeled,'pamap2':di.build_pamap2_labled,
              'uci-har':di.build_uci_har_labeled}
build_dict_un = {'opportunity': di.build_opportunity_unlabeled,'mhealthy':di.build_mhealth_unlabeled,'pamap2':di.build_pamap2_unlabled,
              'uci-har':di.build_uci_har_unlabeled}

build_image = {'opportunity': mk.bulid_markov_transition, 'mhealthy': mk.bulid_markov_transition,'pamap2': mk.bulid_markov_transition,
               'uci-har':mk.bulid_markov_transition}

def load_dataset(dataset, bins, images, seq_length, gyro, preprocess):
    ds = di.load_saved_data(dataset_name=dataset, bins=bins, images=images,seq_length=seq_length, gyro=gyro, preprocess=preprocess)
    if ds is None:
        build_dict_la[dataset](seq_length=seq_length, dataset_name=dataset, bins=bins, images=images)
        build_dict_un[dataset](seq_length=seq_length, dataset_name=dataset, bins=bins, images=images)
        ds = di.load_saved_data(dataset_name=dataset, bins=bins, images=images, seq_length=seq_length, gyro=gyro, preprocess=preprocess)
    return ds

def load_image(dataset,filename,data,image_size,n_bins):
    im = mk.load_saved_image(dataset,filename,n_bins, image_size)
    if im is None:
        build_image[dataset](data,image_size,n_bins,dataset,filename)
        im = mk.load_saved_image(dataset,filename,n_bins, image_size)
    return im
