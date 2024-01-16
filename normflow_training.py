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
from torch.utils.data import Dataset, DataLoader, TensorDataset
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
from time import sleep
import corner
import ray
from ray import air, tune
from ray.air import session
from ray.air.checkpoint import Checkpoint
from ray.tune.schedulers import ASHAScheduler
from os.path import exists

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

num_repeats = 50
num_channels = 3
num_points = 121
in_features = num_points
data_dir = '/nobackup/users/mmdesai/csv_files/'

# functions

def load_in_data(data_dir, name, csv_no, num_points=num_points, num_repeats=num_repeats):
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

def repeated_df_to_tensor(df_varied, df_fixed, batches):
    data_shifted_list = []
    data_unshifted_list = []
    param_shifted_list = []
    param_unshifted_list = []
    for idx in tqdm(range(0, batches)):
        data_shifted = torch.tensor(df_varied.loc[df_varied['batch_id'] == idx].iloc[:, 1:4].values.reshape(num_repeats, num_points, num_channels), dtype=torch.float32).transpose(1, 2)
        data_unshifted = torch.tensor(df_fixed.loc[df_fixed['batch_id'] == idx].iloc[:, 1:4].values.reshape(num_repeats, num_points, num_channels), dtype=torch.float32).transpose(1, 2)
        param_shifted = torch.tensor(df_varied.loc[df_varied['batch_id'] == idx].iloc[::num_points, 5:9].values, dtype=torch.float32).unsqueeze(2).transpose(1,2)
        param_unshifted = torch.tensor(df_fixed.loc[df_fixed['batch_id'] == idx].iloc[::num_points, 5:9].values, dtype=torch.float32).unsqueeze(2).transpose(1,2)
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
        gr_color = self.data_shifted_paper[idx][:, 0, :] - self.data_shifted_paper[idx][:, 1, :]
        gi_color = self.data_shifted_paper[idx][:, 0, :] - self.data_shifted_paper[idx][:, 2, :]
        ri_color = self.data_shifted_paper[idx][:, 1, :] - self.data_shifted_paper[idx][:, 2, :]
        color_shifted_paper = torch.stack((gr_color, gi_color, ri_color), dim = 1)
        gr_color_fix = self.data_unshifted_paper[idx][:, 0, :] - self.data_unshifted_paper[idx][:, 1, :]
        gi_color_fix = self.data_unshifted_paper[idx][:, 0, :] - self.data_unshifted_paper[idx][:, 2, :]
        ri_color_fix = self.data_unshifted_paper[idx][:, 1, :] - self.data_unshifted_paper[idx][:, 2, :]
        color_unshifted_paper = torch.stack((gr_color_fix, gi_color_fix, ri_color_fix), dim = 1)
        return (
            self.param_shifted_paper[idx],
            self.param_unshifted_paper[idx],
            self.data_shifted_paper[idx],
            self.data_unshifted_paper[idx]
        )

def data_wrapper(data_dir='/nobackup/users/mmdesai/csv_files/'):
    print('started dataloading')
    num_csvfiles = 10
    fixed_df = load_in_data(data_dir, 'fixed', num_csvfiles)
    print('got fixed data')
    varied_df = load_in_data(data_dir, 'varied', num_csvfiles)
    print('got varied data')
    num_batches_paper_sample = len(varied_df['batch_id'].unique())
    data_shifted_paper, data_unshifted_paper, param_shifted_paper, param_unshifted_paper = repeated_df_to_tensor(varied_df, fixed_df, num_batches_paper_sample)
    print('finished getting tensors')
    dataset_paper = Paper_data(data_shifted_paper, data_unshifted_paper,
                               param_shifted_paper, param_unshifted_paper,
                               num_batches_paper_sample)
    print(num_batches_paper_sample)
    train_set_size_paper = int(0.8 * num_batches_paper_sample)
    val_set_size_paper = int(0.1 * num_batches_paper_sample)
    test_set_size_paper = num_batches_paper_sample - train_set_size_paper - val_set_size_paper
    print(train_set_size_paper + val_set_size_paper + test_set_size_paper)
    train_data_paper, val_data_paper, test_data_paper = torch.utils.data.random_split(dataset_paper, [train_set_size_paper, val_set_size_paper, test_set_size_paper])
    print('split the data')
    return train_data_paper, val_data_paper, test_data_paper

def prepare_data():
    train_data_paper, val_data_paper, test_data_paper = data_wrapper(data_dir='/nobackup/users/mmdesai/csv_files/')
    torch.save(train_data_paper, "./train_data_paper.pt")
    torch.save(val_data_paper, "./val_data_paper.pt")
    print(type(train_data_paper))
    return test_data_paper

class VICRegLoss(nn.Module):
    def forward(self, x, y, wt_repr=1.0, wt_cov=1.0, wt_std=1.0):
        repr_loss = F.mse_loss(x, y)

        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)
        N = x.size(0)
        D = x.size(-1)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        x = (x-x.mean(dim=0))/std_x
        y = (y-y.mean(dim=0))/std_y

        # transpose dims 1 and 2; keep batch dim i.e. 0, unchanged
        cov_x = (x.transpose(1, 2) @ x) / (N - 1)
        cov_y = (y.transpose(1, 2) @ y) / (N - 1)
        cov_loss = self.off_diagonal(cov_x).pow_(2).sum().div(D)
        cov_loss += self.off_diagonal(cov_y).pow_(2).sum().div(D)

        s = wt_repr*repr_loss + wt_cov*cov_loss + wt_std*std_loss
        return s, repr_loss, cov_loss, std_loss

    def off_diagonal(self,x):
        num_batch, n, m = x.shape
        assert n == m
        # All off diagonal elements from complete batch flattened
        return x.flatten(start_dim=1)[...,:-1].view(num_batch, n - 1, n + 1)[...,1:].flatten()

class ConvResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size, activation=F.relu, dropout_probability=0.1, use_batch_norm=True, zero_initialization=True):
        super().__init__()
        self.activation = activation
        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.batch_norm_layers = nn.ModuleList([nn.BatchNorm1d(channels, eps=1e-3) for _ in range(2)])
        self.conv_layers = nn.ModuleList([nn.Conv1d(channels, channels, kernel_size=kernel_size, padding='same') for _ in range(2)])
        self.dropout = nn.Dropout(p=dropout_probability)
        if zero_initialization:
            nn.init.uniform_(self.conv_layers[-1].weight, -1e-3, 1e-3)
            nn.init.uniform_(self.conv_layers[-1].bias, -1e-3, 1e-3)

    def forward(self, inputs):
        temps = inputs
        if self.use_batch_norm:
            temps = self.batch_norm_layers[0](temps)
        temps = self.activation(temps)
        temps = self.conv_layers[0](temps)
        if self.use_batch_norm:
            temps = self.batch_norm_layers[1](temps)
        temps = self.activation(temps)
        temps = self.dropout(temps)
        temps = self.conv_layers[1](temps)
        return inputs + temps

class ConvResidualNet(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, num_blocks, kernel_size, activation=F.relu, dropout_probability=0.1, use_batch_norm=True):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.initial_layer = nn.Conv1d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=kernel_size, padding='same')
        self.blocks = nn.ModuleList([ConvResidualBlock(channels=hidden_channels, activation=activation, dropout_probability=dropout_probability, use_batch_norm=use_batch_norm, kernel_size=kernel_size) for _ in range(num_blocks)])
        self.final_layer = nn.Conv1d(hidden_channels, out_channels, kernel_size=kernel_size, padding='same')

    def forward(self, inputs):
        temps = self.initial_layer(inputs)
        for block in self.blocks:
            temps = block(temps)
        outputs = self.final_layer(temps)
        return outputs

class SimilarityEmbedding(nn.Module):
    """Simple Dense embedding"""
    def __init__(self, num_dim=3, num_hidden_layers_f=1, num_hidden_layers_h=1, num_blocks=4, kernel_size=5, num_dim_final=10, activation=torch.tanh):
        super(SimilarityEmbedding, self).__init__()
        self.layer_norm = nn.LayerNorm([num_channels, num_points])
        self.num_hidden_layers_f = num_hidden_layers_f
        self.num_hidden_layers_h = num_hidden_layers_h
        self.layers_f = ConvResidualNet(in_channels=num_channels, out_channels=1, hidden_channels=20, num_blocks=num_blocks, kernel_size=kernel_size)
        # self.dropout_layer = nn.Dropout(p=0.2)
        self.contraction_layer = nn.Linear(in_features=in_features, out_features=num_dim)
        self.expander_layer = nn.Linear(num_dim, 20)
        self.layers_h = nn.ModuleList([nn.Linear(20, 20) for _ in range(num_hidden_layers_h)])
        self.final_layer = nn.Linear(20, num_dim_final)
        self.activation = activation

    def forward(self, x):
        x = self.layers_f(x)
        # x = self.dropout_layer(x)
        x = self.contraction_layer(x)
        representation = torch.clone(x)
        x = self.activation(self.expander_layer(x))
        for layer in self.layers_h:
            x = layer(x)
            x = self.activation(x)
        x = self.final_layer(x)

        return x, representation

class EmbeddingNet(nn.Module):
    """Wrapper around the similarity embedding defined above"""
    def __init__(self, num_dim, similarity_embedding, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.representation_net = similarity_embedding
        self.representation_net.load_state_dict(similarity_embedding.state_dict())
        self.num_dim = num_dim
        # the expander network is unused and hence don't track gradients
        for name, param in self.representation_net.named_parameters():
            if 'expander_layer' in name or 'layers_h' in name:
                param.requires_grad = False
        # set freeze status of part of the conv layer of embedding_net
            elif 'layers_f' in name:
                param.requires_grad = True
            else: 
                param.requires_grad = True
                
        self.context_layer = nn.Sequential(
            nn.Linear(num_dim, 1000),
            nn.LeakyReLU(),
            nn.Linear(1000, num_dim)
        )
    def forward(self, x):
        batch_size, channels, dims = x.shape  # 5000, 3, 121
        _, rep = self.representation_net(x)
        rep = rep.reshape(batch_size, self.num_dim)  # squeezes out the middle "channel" dimension

        return self.context_layer(rep)

def normflow(similarity_embedding, num_transforms, num_blocks, hidden_features, context_features, num_dim):
    base_dist = StandardNormal([3])
    transforms = []
    features = num_dim
    for _ in range(num_transforms):
        block = [MaskedAffineAutoregressiveTransform(
                features=features,  # 2-dim posterior
                hidden_features=hidden_features,
                context_features=context_features,  # 2
                num_blocks=num_blocks,
                activation=torch.tanh,
                use_batch_norm=False,
                use_residual_blocks=True,
                dropout_probability=0.01,
    #             integrand_net_layers=[20, 20]
            ),
            RandomPermutation(features=features)
        ]
        transforms += block
    transform = CompositeTransform(transforms)
    embedding_net = EmbeddingNet(num_dim, similarity_embedding)
    flow = Flow(transform, base_dist, embedding_net).to(device=device)
    return flow

def train_one_epoch(epoch_index, data_loader, flow, optimizer):
    running_loss = 0.
    last_loss = 0.
    for idx, val in enumerate(data_loader, 1):
        augmented_shift, unshifted_shift, augmented_data, unshifted_data = val
        augmented_shift = augmented_shift[...,0:3].to(device)
        augmented_shift = augmented_shift.flatten(0, 2).to(device)
        augmented_data = augmented_data.reshape(-1, 3, num_points).to(device)
        loss = 0
        flow_loss = -flow.log_prob(augmented_shift, context=augmented_data).mean()
        optimizer.zero_grad()
        flow_loss.backward()
        optimizer.step()
        loss += flow_loss.item()
        running_loss += loss
        n = 10
        if idx % n == 0:
            last_loss = running_loss / n
            print(' Avg. train loss/batch after {} batches = {:.4f}'.format(idx, last_loss))
            running_loss = 0.
    return last_loss

def val_one_epoch(epoch_index, data_loader, flow):
    running_loss = 0.
    last_loss = 0.
    for idx, val in enumerate(data_loader, 1):
        augmented_shift, unshifted_shift, augmented_data, unshifted_data = val
        augmented_shift = augmented_shift[...,0:3].to(device)
        augmented_shift = augmented_shift.flatten(0, 2).to(device)
        augmented_data = augmented_data.reshape(-1, 3, num_points).to(device)
        loss = 0
        flow_loss = -flow.log_prob(augmented_shift, context=augmented_data).mean()
        loss += flow_loss.item()
        running_loss += loss
        n = 1
        if idx % n == 0:
            last_loss = running_loss / n
            print(' Avg. train loss/batch after {} batches = {:.4f}'.format(idx, last_loss))
            running_loss = 0.
    return last_loss

def train_flow(config):
    print('starting training')
    # get the best similarity embedding
    LOADPATH = '/nobackup/users/mmdesai/bestembeddingweights/similarity-embedding-weights.pth'
    model_state = torch.load(LOADPATH)
    similarity_embedding = SimilarityEmbedding(num_dim = 3, num_hidden_layers_f=4, num_hidden_layers_h=3, num_blocks=5, kernel_size=9, num_dim_final=10).to(device)
    num_dim = 3
    similarity_embedding.load_state_dict(model_state)
    print(similarity_embedding.eval())
    # dataset
    data_shifted_paper = torch.load('/nobackup/users/mmdesai/newtensors/data_shifted_paper.pt')
    data_unshifted_paper = torch.load('/nobackup/users/mmdesai/newtensors/data_unshifted_paper.pt')
    param_shifted_paper = torch.load('/nobackup/users/mmdesai/newtensors/param_shifted_paper.pt')
    param_unshifted_paper = torch.load('/nobackup/users/mmdesai/newtensors/param_unshifted_paper.pt')
    print('finished getting tensors')
    num_batches_paper_sample = 9989
    dataset_paper = Paper_data(data_shifted_paper, data_unshifted_paper, param_shifted_paper, param_unshifted_paper, num_batches_paper_sample)
    print('dataset defined')
    train_set_size_paper = int(0.8 * num_batches_paper_sample)
    val_set_size_paper = int(0.1 * num_batches_paper_sample)
    test_set_size_paper = num_batches_paper_sample - train_set_size_paper - val_set_size_paper
    train_data_paper, val_data_paper, test_data_paper = torch.utils.data.random_split(dataset_paper, [train_set_size_paper, val_set_size_paper, test_set_size_paper])
    train_data_loader_paper = DataLoader(train_data_paper, int(config["batch_size"]), shuffle=True, num_workers=4)
    val_data_loader_paper = DataLoader(val_data_paper, int(config["batch_size"]), shuffle=True, num_workers=4)
    print('dataset loaded')
    for var_inj_se, fix_inj_se, var_data_se, fix_data_se in train_data_loader_paper:
        var_inj_se = var_inj_se.reshape((-1,)+var_inj_se.shape[2:]).to(device)
        fix_inj_se = fix_inj_se.reshape((-1,)+fix_inj_se.shape[2:]).to(device)
        var_data_se = var_data_se.reshape((-1,)+var_data_se.shape[2:]).to(device)
        fix_data_se = fix_data_se.reshape((-1,)+fix_data_se.shape[2:]).to(device)
        break
    print(var_data_se.shape)
    # define the flow
    _, rep = similarity_embedding(var_data_se)  # _.shape = batch_size x 1 x 10, # rep.shape = batch_size x 1 x 2
    context_features = rep.shape[-1]
    flow = normflow(similarity_embedding, config['num_transforms'], config['num_blocks'], config['hidden_features'], context_features=context_features, num_dim=num_dim)
    print('Total number of trainable parameters: ', sum(p.numel() for p in flow.parameters() if p.requires_grad))
    # optimizer
    optimizer = optim.SGD(flow.parameters(), lr=config['lr'], momentum=0.5)
    # scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1, threshold=0.001)
    # set checkpoint
    loaded_checkpoint = session.get_checkpoint()
    if loaded_checkpoint:
        with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
           flow_state, optimizer_state = torch.load(os.path.join(loaded_checkpoint_dir, "checkpoint.pt"))
        flow.load_state_dict(flow_state)
        optimizer.load_state_dict(optimizer_state)
    print('made checkpoint')
    # train
    EPOCHS = 100
    epoch_number = 0
    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))
        flow.train(True)
        for name, param in flow._embedding_net.named_parameters():
            param.requires_grad = True
        avg_train_loss = train_one_epoch(epoch_number, train_data_loader_paper, flow, optimizer)
        flow.train(False)
        avg_val_loss = val_one_epoch(epoch_number, val_data_loader_paper, flow)
        print(f"Train/Val flow Loss after epoch: {avg_train_loss:.4f}/{avg_val_loss:.4f}")
        epoch_number += 1
        scheduler.step(avg_val_loss)
        for param_group in optimizer.param_groups:
            print("Current LR = {:.3e}".format(param_group['lr']))
        os.makedirs("flow_model", exist_ok=True)
        torch.save((flow.state_dict(), optimizer.state_dict()), "flow_model/checkpoint.pt")
        checkpoint = Checkpoint.from_directory("flow_model")
        session.report({"avg_train_loss":avg_train_loss, "avg_val_loss":avg_val_loss}, checkpoint=checkpoint)
    print("Finished Training")


def test_best_model(best_result):
    best_trained_model = normflow(best_result.config['num_transforms'], best_result.config['num_blocks'], best_result.config['hidden_features'], context_features)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    best_trained_model.to(device)
    checkpoint_path = os.path.join(best_result.checkpoint.to_directory(), "checkpoint.pt")
    flow_state, optimizer_state = torch.load(checkpoint_path)
    best_trained_model.load_state_dict(flow_state)
    model_state, optimizer_state = torch.load(checkpoint_path)
    SAVEPATH = '/nobackup/users/mmdesai/bestflowweights/flow-weights.pth'
    torch.save(flow_state, SAVEPATH)
    print(best_trained_model.eval())
    LOADPATH = '/nobackup/users/mmdesai/train_se_2024-01-05_08-05-12/train_se_10fe8_00005_5_batch_size=300,kernel_size=9,lr=0.0001,num_blocks=5,num_dim=3,num_dim_final=20,num_hidden_layers_f=4,num_hi_2024-01-05_08-05-15/my_model'
    model_state, optimizer_state = torch.load(LOADPATH + '/checkpoint.pt')
    similarity_embedding = SimilarityEmbedding(num_dim=3, num_hidden_layers_f=4, num_hidden_layers_h=2, num_blocks=5, kernel_size=9, num_dim_final=20).to(device)
    similarity_embedding.load_state_dict(model_state)
    print(similarity_embedding.eval())
    data_shifted_paper = torch.load('/nobackup/users/mmdesai/newtensors/data_shifted_paper.pt')
    data_unshifted_paper = torch.load('/nobackup/users/mmdesai/newtensors/data_unshifted_paper.pt')
    param_shifted_paper = torch.load('/nobackup/users/mmdesai/newtensors/param_shifted_paper.pt')
    param_unshifted_paper = torch.load('/nobackup/users/mmdesai/newtensors/param_unshifted_paper.pt')
    num_batches_paper_sample = 9989
    dataset_paper = Paper_data(data_shifted_paper, data_unshifted_paper, param_shifted_paper, param_unshifted_paper)
    train_set_size_paper = int(0.8 * num_batches_paper_sample)
    val_set_size_paper = int(0.1 * num_batches_paper_sample)
    test_set_size_paper = num_batches_paper_sample - train_set_size_paper - val_set_size_paper
    train_data_paper, val_data_paper, test_data_paper = torch.utils.data.random_split(dataset_paper, [train_set_size_paper, val_set_size_paper, test_set_size_paper])
    test_data_loader_paper = DataLoader(test_data_paper, batch_size=1, shuffle=False, num_workers=2)
    for var_inj_se, fix_inj_se, var_data_se, fix_data_se in train_data_loader_paper:
        var_inj_se = var_inj_se.reshape((-1,)+var_inj_se.shape[2:]).to(device)
        fix_inj_se = fix_inj_se.reshape((-1,)+fix_inj_se.shape[2:]).to(device)
        var_data_se = var_data_se.reshape((-1,)+var_data_se.shape[2:]).to(device)
        fix_data_se = fix_data_se.reshape((-1,)+fix_data_se.shape[2:]).to(device)
        break
    print(var_data_se.shape)
    # define the flow
    _, rep = similarity_embedding(var_data_se)  # _.shape = batch_size x 1 x 10, # rep.shape = batch_size x 1 x 2
    context_features = rep.shape[-1]
    num_dim = 3
    for idx, (_, shift_test, data_test, data_test_orig) in enumerate(test_data_loader_paper):
        augmented_shift, unshifted_shift, augmented_data, unshifted_data = val
        augmented_shift = augmented_shift[...,0:3].to(device)
        augmented_shift = augmented_shift.flatten(0, 2).to(device)
        augmented_data = augmented_data.reshape(-1, 3, num_points).to(device)
        with torch.no_grad():
            test_loss = -flow.log_prob(augmented_shift, context=augmented_data).mean()
    print('Best trial test loss: {}'.format(test_loss))


def main(num_samples=10, max_num_epochs=50, gpus_per_trial=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    config = {
        'num_transforms': tune.choice([3, 4, 5, 6, 7, 8]),
        'num_blocks': tune.choice([3, 4, 5, 6, 7, 8]),
        'hidden_features': tune.choice([40, 50, 60, 70, 80, 90]),
        "lr": tune.loguniform(1e-5, 1e-3),
        "batch_size": tune.choice([100, 200, 300, 400, 500]),
    }
    tune_scheduler = ASHAScheduler(
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train_flow),
            resources={"cpu": 5, "gpu": gpus_per_trial}  # TBD: replace 2 by varname
        ),
        tune_config=tune.TuneConfig(
            metric="avg_val_loss",
            mode="min",
            scheduler=tune_scheduler,
            num_samples=num_samples,
        ),
        param_space=config,
        run_config=air.RunConfig(storage_path="./")
    )
    results = tuner.fit()
    best_result = results.get_best_result("avg_val_loss", "min")
    print("Best trial config: {}".format(best_result.config))
    print("Best trial final train loss: {}".format(
        best_result.metrics["avg_train_loss"]))
    print("Best trial final validation loss: {}".format(
        best_result.metrics["avg_val_loss"]))
    
    test_best_model(best_result)

main(num_samples=30, max_num_epochs=100, gpus_per_trial=1)



