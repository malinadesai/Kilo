import numpy as np
import pandas as pd
import os, sys, time, glob
import json
import warnings
from tqdm import tqdm
import nflows.utils as torchutils
from IPython.display import clear_output
from time import time
from time import sleep
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
from os.path import exists

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def open_json(
    file_name, dir_path
):
    ''' 
    Opens a json file, loads data as a dictionary, and closes the file 
    Inputs:
        file_name: /name of json file.json
        dir_path: directory containing json files 
    Outputs:
        data: dictionary containing json content
    '''
    f = open(dir_path + file_name)
    data = json.load(f)
    f.close()
    return data

def get_names(
    path, label, set, num
):
    ''' 
    Gets the file path for the fixed data
    Inputs:
        path: string, directory to point to
        label: string, label assigned during nmma light curve generation
        set: int, number in directory name
        num: int, number of files to unpack
    Outputs: 
        file_names: list, contains full path file names
    '''
    file_names = [0] * num
    for i in range(0, num):
        one_name = path + '/{}_batch_{}/{}_{}_{}.json'.format(
            label, set, label, set, i
        )
        file_names[i] = one_name
    return file_names

def json_to_df(
    file_name, dir_path, detection_limit, bands
):
    ''' 
    Flattens a light curve json file into a DataFrame
    Inputs:
        file_name: light curve json file name
        detection_limit: float, photometric detection limit
        bands: list, contains the json photometry keys as strings
    Outputs:
        df_unpacked: DataFrame containing the photometry data, time, 
                      and number of total detections across all bands
    '''
    data = open_json(file_name, dir_path)
    df = pd.DataFrame.from_dict(data, orient="columns")
    df_unpacked = pd.DataFrame(columns=bands)
    counter = 0
    for j in range(len(bands)):
        df_unpacked[['t', bands[j], 'x']] = pd.DataFrame(
            df[bands[j]].tolist(), index= df.index
        )
        for val in df_unpacked[bands[j]]:
            if val != detection_limit:
                counter += 1
            else:
                pass
    df_unpacked['num_detections'] = np.full(len(df_unpacked), counter)
    df_unpacked = df_unpacked.drop(columns=['x'])
    return df_unpacked

def extract_number(
    file_name
):
    '''
    Gets the number in a file name
    Inputs:
        file_name: string, name of file that has a number
    Outputs:
        Number in file name, or inf if none
    '''
    try:
        return int("".join(filter(str.isdigit, file_name)))
    except ValueError:
        return float("inf")

def directory_json_to_df(
    dir_path, label, detection_limit, bands
):
    '''
    Takes a directory of light curves and converts them to DataFrames
    Inputs:
        dir_path: directory containing light curve files
        label: string, label used when generating the light curve data
        detection_limit: float, photometric detection limit
        bands: list, contains the json photometry keys as strings
    Outputs:
        df_list: list, contains all files as DataFrames
    '''
    df_list = []
    for file in sorted(os.listdir(dir_path), key=extract_number):
        if file.endswith(".json") and file.startswith(label):
            df = json_to_df(
                file, dir_path, detection_limit, bands
            )
            df['sim_id'] = extract_number(file)
            df_list.append(df)
    return df_list

def pad_the_data(df, t_min, t_max, step, data_filler, bands):
    '''
    Takes DataFrames and adds filler values to both ends, preserves 
    original time information.
    Inputs:
        df: DataFrame with 't' and photometric columns 
        t_min: float, global minimum start time
        t_max: float, global maximum end time
        step: float, time step between rows
        data_filler: value to use in the filler rows
        bands: list of photometric columns to fill
                (e.g. ['ztfg', 'ztfr', 'ztfi'])
    Outputs:
        df_padded: DataFrame with original data and padded rows,
                    covering full time range
    '''
    df = df.copy()
    df = df.sort_values('t').reset_index(drop=True)
    full_time = np.arange(t_min, t_max, step)
    num_points = len(full_time)
    t_start = df['t'].min()
    t_end = df['t'].max()
    prepend_times = full_time[full_time < t_start]
    append_times = full_time[full_time > t_end]

    def make_filler(times):
        return pd.DataFrame({
            't': times,
            **{f: data_filler for f in bands},
            'sim_id': np.nan,
            'num_detections': np.nan
        })

    prepend_df = make_filler(prepend_times)
    append_df = make_filler(append_times)
    df_padded = pd.concat([prepend_df, df, append_df], ignore_index=True)
    df_padded = df_padded.sort_values('t').reset_index(drop=True)
    assert np.isclose(df_padded['t'].min(), t_min), \
    f"Start time is {df_padded['t'].min()}, expected {t_min}"
    assert np.isclose(df_padded['t'].max(), t_max - step), \
    f"End time is {df_padded['t'].max()}, expected {t_max - step}"
    try:
        assert len(df_padded) == num_points+1, \
        f"Length is {len(df_padded)}, expected {num_points+1}"
    except AssertionError as e:
        count = num_points + 1 - len(df_padded)
        addt_times = np.arange(t_max, t_max+(step*count), step)
        addt_append = make_filler(addt_times)
        df_padded = pd.concat([df_padded, addt_append], ignore_index=True)
        df_padded = df_padded.sort_values('t').reset_index(drop=True)
        assert len(df_padded) == num_points+1, \
        f"Length is {len(df_padded)}, expected {num_points+1}"
    return df_padded

def pad_all_dfs(
    df_list, t_min, t_max, step, data_filler, bands
):
    '''
    Pads multiple DataFrames at a time
    Inputs: 
        df_list: list of DataFrames to pad
    Outputs:
        padded_df_list: list of DataFrames after padding
    '''
    padded_df_list = []
    for i in tqdm(range(len(df_list))):
        df = df_list[i]
        sim_num = df.iloc[0, df.columns.get_loc('sim_id')]
        det_num = df.iloc[0, df.columns.get_loc('num_detections')]
        df = pad_the_data(
            df, t_min, t_max, step, data_filler, bands
        )
        df['sim_id'] = np.full(len(df), sim_num)
        df['num_detections'] = np.full(len(df), det_num)
        padded_df_list.append(df)
    return padded_df_list

def find_min_max_t(
    df_list
):
    '''
    Finds the maximum and minimum time of all provided DataFrames 
    Inputs:
        df_list: list of DataFrames
    Outputs:
        t_min: float, minimum time across all DataFrames
        t_max: float, maximum time across all DataFrames
    '''
    t_mins = []
    t_maxs = []
    for i in range(len(df_list)):
        t_mins.append(df_list[i]['t'].min())
        t_maxs.append(df_list[i]['t'].max())
    t_min = min(t_mins)
    t_max = max(t_maxs)
    return t_min, t_max

def grab_injection(
    inj_file, dir_path
):
    '''
    Reads in the injection file
    Inputs:
        inj_file: string, injection file name
        dir_path: string, directory path
    Outputs:
        inj_df: DataFrame containing the injection parameters
    '''
    data = open_json(inj_file, dir_path)
    content = data['injections']['content']
    inj_df = pd.DataFrame.from_dict(content)
    return inj_df

def load_light_curves_df(
    dir_path, 
    inj_file, 
    label, 
    detection_limit, 
    bands, 
    step, 
    data_filler,
    num_repeats=1,
    add_batch_id=False,
):
    '''
    Converts NMMA generated light curves to a DataFrame
    Inputs:
        dir_path: string, directory path
        inj_file: string, injection file name
        label: string, label assigned during nmma light curve generation
        detection_limit: float, photometric detection limit
        bands: list of photometric columns to fill
               (e.g. ['ztfg', 'ztfr', 'ztfi'])
        step: float, time step between rows
        data_filler: float, value to use in the filler rows
    Outputs:
        lc_df: DataFrame, contains light curve and injection data
    '''
    df_list = directory_json_to_df(
        dir_path=dir_path, 
        label=label, 
        detection_limit=detection_limit, 
        bands=bands)
    t_min, t_max = find_min_max_t(df_list)
    num_points = len(np.arange(t_min, t_max, step)) + 1
    padded_list = pad_all_dfs(
        df_list, 
        t_min=t_min, 
        t_max=t_max, 
        step=step, 
        data_filler=data_filler, 
        bands=bands)
    all_padded_lcs = pd.concat(padded_list).reset_index(drop=True)
    inj_df = grab_injection(inj_file=inj_file, dir_path=dir_path)
    lc_df = all_padded_lcs.merge(inj_df, on='simulation_id')
    if num_repeats <= 0:
        print('Warning: num_repeats must be at least 1 (for one lc!).' + 
              'Defaulting to 1.')
        num_repeats = 1
    if add_batch_id == True:
        lc_df['batch_id'] = lc_df.index // (num_points * num_repeats)
    return lc_df

def load_in_data(data_dir, name, csv_no, num_points=num_points, num_repeats=num_repeats):
    '''
    Loading in multiple saved csv files containing light curve data as one dataframe
    Inputs:
        data_dir: directory containing the csv files
        csv_no: number of csv files to load in
        num_points: number of data points per light curve
        num_repeats: repeats of injection parameters to determine batches
    Outputs:
        data_df: single dataframe containing the data 
    '''
    data_list = []
    for i in range (0, csv_no):
        data_list.append(pd.read_csv(data_dir + '{}_{}.csv'.format(name, i)))
    data_df = pd.concat(data_list)
    num_sims = int(len(data_df)/num_points)
    sim_list = []
    sim_no = 0
    for i in range(0, num_sims):
        for j in range(0, num_points):
            sim_list.append(sim_no)
        sim_no += 1
    data_df['sim_id'] = sim_list
    batch_list = []
    batch_no = 0
    num_batches = int((len(data_df)/num_points)/num_repeats)
    data_df = data_df.iloc[0:(num_batches*num_points*num_repeats), :].copy()
    for i in range(0, num_batches):
        for j in range(0, num_points*num_repeats):
            batch_list.append(batch_no)
        batch_no += 1
    data_df['batch_id'] = batch_list
    return data_df
    
def match_fix_to_var(data_dir, name1, name2, start, stop, num_points=num_points, num_repeats=num_repeats):
    '''
    Matches the shifted injection light curve data to its fixed counterpart
    Inputs:
        data_dir: directory containing the data
        name1: label of the varied csv files
        name2: label of the fixed csv files
        start: starting csv number
        stop: ending csv number
        num_points: number of points in the light curve
        num_repeats: number of repeated mass, velocity, lanthanide injections
    Outputs:
        fixed_data_df: returns the fixed portion of the light curve data
        varied_data_df: returns the shifted/varied portion of the light curve data
    '''
    # initiate list for dataframes
    fixed_list = []
    varied_list = []
    # do all data processing for a given number of dataframes
    for i in range (start, stop):
        # load in the data
        df_var = pd.read_csv(data_dir + '{}_{}.csv'.format(name1, i))
        df_fix = pd.read_csv(data_dir + '{}_{}.csv'.format(name2, i))
        # match the two dataframes to each other based on sim id
        matched = df_var.merge(df_fix, left_on=['sim_id', df_var.groupby('sim_id').cumcount()],
                               right_on=['sim_id', df_fix.groupby('sim_id').cumcount()])
        # grab the fixed and varied portions of the dataframe
        fix_df = matched.iloc[:, 12:]
        var_df = matched.iloc[:, :12]
        # adjust columns and column names
        fix_df.columns = fix_df.columns.str.rstrip('_y')
        var_df.columns = var_df.columns.str.rstrip('_x')
        var_df = var_df.drop(columns=['key_1'])
        # add to list of dataframes
        fixed_list.append(fix_df)
        varied_list.append(var_df)
    # concatenate the list of dataframes together
    fixed_data_df = pd.concat(fixed_list)
    varied_data_df = pd.concat(varied_list)
    # overwrite the simulation id's and add batch id's
    num_sims = int(len(fixed_data_df)/num_points)
    sim_list = []
    sim_no = 0
    for i in range(0, num_sims):
        for j in range(0, num_points):
            sim_list.append(sim_no)
        sim_no += 1
    fixed_data_df['sim_id'] = sim_list
    varied_data_df['sim_id'] = sim_list
    batch_list = []
    batch_no = 0
    num_batches = int((len(fixed_data_df)/num_points)/num_repeats)
    fixed_data_df = fixed_data_df.iloc[0:(num_batches*num_points*num_repeats), :].copy()
    varied_data_df = varied_data_df.iloc[0:(num_batches*num_points*num_repeats), :].copy()
    for i in range(0, num_batches):
        for j in range(0, num_points*num_repeats):
            batch_list.append(batch_no)
        batch_no += 1
    fixed_data_df['batch_id'] = batch_list
    varied_data_df['batch_id'] = batch_list
    
    return fixed_data_df, varied_data_df

def matched(data_dir, name1, name2, start, stop, num_points=num_points, num_repeats=num_repeats):
    '''
    Matches light curves with the same injection parameters
    Inputs:
        data_dir: file path for data
        name1: label of the varied csv files
        name2: label of the fixed csv files
        start: starting csv number
        stop: ending csv number
        num_points: number of points in the light curve
        num_repeats: number of repeated mass, velocity, lanthanide injections
    Outputs:
        matched_df: combined dataframe of the shifted and fixed light curves
    '''
    # initiate list for dataframes
    matched_list = []
    # do all data processing for a given number of dataframes
    for i in range (start, stop):
        # load in the data
        df_var = pd.read_csv(data_dir + '{}_{}.csv'.format(name1, i))
        df_fix = pd.read_csv(data_dir + '{}_{}.csv'.format(name2, i))
        # match the two dataframes to each other based on sim id
        matched = df_var.merge(df_fix, left_on=['sim_id', df_var.groupby('sim_id').cumcount()],
                               right_on=['sim_id', df_fix.groupby('sim_id').cumcount()])
        matched_list.append(matched)
    matched_df = pd.concat(matched_list)
    return matched_df

def add_batch_sim_nums_all(df, num_points=num_points, num_repeats=num_repeats):
    '''
    Adds a simulation and batch id number to each light curve
    Inputs:
        df: dataframe containing light curve data
        num_points: number of points in the light curve
        num_repeats: number of repeated mass, velocity, lanthanide injections
    Outputs:
        None
    '''
    num_batches_split = int((len(df)/num_points)/num_repeats)
    batch_list_split = []
    batch_no = 0
    for i in range(0, num_batches_split):
        for j in range(0, num_repeats*num_points):
            batch_list_split.append(batch_no)
        batch_no += 1
    df['batch_id'] = batch_list_split

    num_sims_split = int(len(df)/num_points)
    sim_list_split = []
    sim_no = 0
    for i in range(0, num_sims_split):
        for j in range(0, num_points):
            sim_list_split.append(sim_no)
        sim_no += 1
    df['sim_id'] = sim_list_split

def get_test_names(path, label, set, num):
    ''' 
    Gets the file path for the fixed data
    Inputs:
        path = string, directory to point to
        label = string, label assigned during nmma light curve generation
        set = int, number in directory name
        num = int, number of files to unpack
    Returns: 
        list, contains full path file names
    '''
    file_names = [0] * num
    for i in range(0, num):
        one_name = path + '/{}{}_{}.json'.format(label, set, i)
        file_names[i] = one_name
    return file_names

def repeated_df_to_tensor(df_varied, df_fixed, batches):
    '''
    Converts dataframes into pytorch tensors
    Inputs:
        df_varied: dataframe containing the shifted light curve information
        df_fixed: dataframe containing the analagous fixed light curve information
        batches: number of unique mass, velocity, and lanthanide injections
    Outputs:
        data_shifted_list: list of tensors of shape [repeats, channels, num_points] containing the shifted light curve photometry
        data_unshifted_list: list of tensors of shape [repeats, channels, num_points] containing the fixed light curve photometry
        param_shifted_list: list of tensors of shape [repeats, 1, 5] containing the injection parameters of the shifted light curves
        param_unshifted_list: list of tensors of shape [repeats, 1, 5] containing the injection parameters of the fixed light curves
    '''
    data_shifted_list = []
    data_unshifted_list = []
    param_shifted_list = []
    param_unshifted_list = []
    for idx in tqdm(range(0, batches)):
        data_shifted = torch.tensor(df_varied.loc[df_varied['batch_id'] == idx].iloc[:, 1:4].values.reshape(num_repeats, num_points, num_channels), 
                                    dtype=torch.float32).transpose(1, 2)
        data_unshifted = torch.tensor(df_fixed.loc[df_fixed['batch_id'] == idx].iloc[:, 1:4].values.reshape(num_repeats, num_points, num_channels), 
                                    dtype=torch.float32).transpose(1, 2)
        param_shifted = torch.tensor(df_varied.loc[df_varied['batch_id'] == idx].iloc[::num_points, 6:11].values, 
                                    dtype=torch.float32).unsqueeze(2).transpose(1,2)
        param_unshifted = torch.tensor(df_fixed.loc[df_fixed['batch_id'] == idx].iloc[::num_points, 5:10].values, 
                                    dtype=torch.float32).unsqueeze(2).transpose(1,2)
        data_shifted_list.append(data_shifted)
        data_unshifted_list.append(data_unshifted)
        param_shifted_list.append(param_shifted)
        param_unshifted_list.append(param_unshifted)
    return data_shifted_list, data_unshifted_list, param_shifted_list, param_unshifted_list

class Paper_data(Dataset):
    def __init__(self, data_shifted_paper, data_unshifted_paper,
                 param_shifted_paper, param_unshifted_paper,
                 num_batches_paper_sample):
        super().__init__()
        self.data_shifted_paper = data_shifted_paper
        self.data_unshifted_paper = data_unshifted_paper
        self.param_shifted_paper = param_shifted_paper
        self.param_unshifted_paper = param_unshifted_paper
        self.num_batches_paper_sample = num_batches_paper_sample

    def __len__(self):
        return self.num_batches_paper_sample

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        return (
            self.param_shifted_paper[idx].to(device),
            self.param_unshifted_paper[idx].to(device),
            self.data_shifted_paper[idx].to(device),
            self.data_unshifted_paper[idx].to(device)
        )
