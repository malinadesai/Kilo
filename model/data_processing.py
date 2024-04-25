import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.lines as mlines
import os, sys, time, glob
import json
import copy
import scipy
import argparse
import warnings
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import bilby
from bilby.core.prior import Uniform, DeltaFunction
from bilby.core.likelihood import GaussianLikelihood
from nflows.nn.nets.resnet import ResidualNet
from nflows import transforms, distributions, flows
from nflows.distributions import StandardNormal
from nflows.flows import Flow
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms import CompositeTransform, RandomPermutation
import nflows.utils as torchutils
from IPython.display import clear_output
from time import time
from time import sleep
import corner
import torchvision
import torchvision.transforms as transforms
from os.path import exists
from resnet import ResNet

bands = ['ztfg', 'ztfr', 'ztfi']
detection_limit = 22.0
num_repeats = 50
num_channels = 3
num_points = 121
in_features = num_points

t_zero = 44242.00021937881
t_min = 44240.00050450478
t_max = 44269.99958898723
days = int(round(t_max - t_min))
time_step = 0.25

def open_json(file_name, dir_path):
    ''' 
    Opens a json file, loads the data as a dictionary, and closes the file 
    Inputs:
        file_name = /name of json file.json
        dir_path = directory containing json files 
    Returns:
        data = dictionary containing json content
    '''
    f = open(dir_path + file_name)
    data = json.load(f)
    f.close()
    return data

def get_names(path, label, set, num):
    ''' 
    Gets the file path for the fixed data
    Inputs:
        path = string, directory to point to
        label = string, label assigned during nmma light curve generation
        set = int, number in directory name
        num = int, number of files to unpack
    Returns: 
        file_names = list, contains full path file names
    '''
    file_names = [0] * num
    for i in range(0, num):
        one_name = path + '/{}_batch_{}/{}_{}_{}.json'.format(label, set, label, set, i)
        file_names[i] = one_name
    return file_names

def json_to_df(file_names, num_sims, detection_limit=detection_limit, bands=bands):
    ''' 
    Flattens json files into a dataframe
    Inputs:
        file_names = list, contains full path file names as strings
        num_sims = int, number of files to unpack
        detection_limit = float, photometric detection limit
        bands = list, contains the json photometry keys as strings
    Returns:
        df_list = list of dataframes containing the photometry data, time, and number of total detections across all bands
    '''
    df_list = [0] * num_sims
    for i in tqdm(range(num_sims)):
        data = json.load(open(file_names[i], "r"))
        df = pd.DataFrame.from_dict(data, orient="columns")
        df_unpacked = pd.DataFrame(columns=bands)
        counter = 0
        for j in range(len(bands)):
            df_unpacked[['t', bands[j], 'x']] = pd.DataFrame(df[bands[j]].tolist(), index= df.index)
            for val in df_unpacked[bands[j]]:
                if val != detection_limit:
                    counter += 1
                else:
                    pass
        df_unpacked['num_detections'] = np.full(len(df_unpacked), counter)
        df_unpacked['sim_id'] = np.full(len(df_unpacked), i)
        df_unpacked = df_unpacked.drop(columns=['x'])
        df_list[i] = df_unpacked
    return df_list

def gen_prepend_filler(data_filler, t_min, t_max, step = 0.25):
    '''
    front end padding
    Inputs:
        data_filler: number that is used as the filler, ie the detection limit
        t_min: minimum time
        t_max: maximum time
        step: time increment
    Outputs:
        filler_df: dataframe to pad the existing data
    '''
    # Fill according to step size, regardless of count
    ar = np.arange(start=t_min, stop=t_max, step=step)
    filler_dict = {'t':ar, 'ztfg':[data_filler]*len(ar), 'ztfr':[data_filler]*len(ar), 'ztfi':[data_filler]*len(ar),
                   'sim_id':[np.nan]*len(ar), 'num_detections':[np.nan]*len(ar)}
    filler_df = pd.DataFrame(filler_dict)
    return filler_df

def gen_append_filler(data_filler, t_min, count, step=0.25):
    '''
    back end padding
    Inputs:
        data_filler: number that is used as the filler, ie the detection limit
        t_min: minimum time
        t_max: maximum time
        step: time increment
    Outputs:
        filler_df: dataframe to pad the existing data
    '''
    # Fill according to min value and specified count
    l = np.arange(start=t_min, stop=t_min+(count*step), step=step)
    filler_dict = {'t':l, 'ztfg':[data_filler]*len(l), 'ztfr':[data_filler]*len(l), 'ztfi':[data_filler]*len(l),
                   'sim_id':[np.nan]*len(l), 'num_detections':[np.nan]*len(l)}
    filler_df = pd.DataFrame(filler_dict)
    return filler_df

def pad_the_data(actual_df, desired_count=num_points, filler_time_step=time_step, filler_data=detection_limit):
    '''
    pads both ends of the light curve dataframe to ensure consistent number of data points across all data
    Inputs:
        actual_df: existing light curve data
        desired_count: desired length of data, ie number of points for the light curve
        filler_time_step: time increment
        filler_data: number that is used as the filler, ie the detection limit
    Outputs:
        cat_df: padded dataframe
    '''
    actual_df.iloc[:, actual_df.columns.get_loc('t')] = actual_df.iloc[:, actual_df.columns.get_loc('t')].apply(lambda x: x - t_min) 
    cat_df = actual_df
    cat_count = len(cat_df)
    prepended_count = 0
    if (actual_df['t'].min() >= filler_time_step):
        filler_max_time = actual_df['t'].min() - filler_time_step  # stop one time step before current min
        prepend_filler_df = gen_prepend_filler(filler_data, 0, filler_max_time, filler_time_step)
        prepended_count = len(prepend_filler_df)
        cat_df = pd.concat([prepend_filler_df, actual_df], ignore_index=True)
        cat_count = len(cat_df)
    append_count = desired_count - cat_count
    if append_count > 0: 
        max_t = cat_df['t'].max() 
        steps_per_count = 1/filler_time_step
        filler_min_time = int(max_t*steps_per_count)/steps_per_count + filler_time_step  # start at next time step
        append_filler_df = gen_append_filler(filler_data, filler_min_time, append_count)
        cat_df = pd.concat([cat_df, append_filler_df], ignore_index=True)
        cat_count = len(cat_df)
    assert(len(cat_df) == desired_count)
    return cat_df





