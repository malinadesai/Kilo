# import modules
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
from IPython.display import clear_output
from time import sleep
from filelock import FileLock
import corner
import torchvision
import torchvision.transforms as transforms
import ray
from ray import air, tune
from ray.air import session
from ray.air.checkpoint import Checkpoint
from ray.tune.schedulers import ASHAScheduler
from os.path import exists
from resnet import ResNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

num_repeats = 50
num_channels = 3
num_points = 121
in_features = num_points

class Paper_data(Dataset):
    '''
    Dataset processing
    '''
    def __init__(self, data_shifted, data_unshifted,
                 param_shifted, param_unshifted, num_batches):
        super().__init__()
        self.data_shifted = data_shifted
        self.data_unshifted = data_unshifted
        self.param_shifted = param_shifted
        self.param_unshifted = param_unshifted
        self.num_batches = num_batches

    def __len__(self):
        return self.num_batches

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return (
            self.param_shifted[idx],
            self.param_unshifted[idx],
            self.data_shifted[idx],
            self.data_unshifted[idx]
        )

class VICRegLoss(nn.Module):
    '''
    Variance-Invariance-Covariance Regularization Loss from https://arxiv.org/abs/2105.04906
    '''
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
        x = x.unsqueeze(1)
        y = y.unsqueeze(1)
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
    '''
    A fully connective neural network with a ResNet layer f  and an expander layer h
    '''
    def __init__(self, num_dim=3, num_hidden_layers_f=1, num_hidden_layers_h=1, num_blocks=4, kernel_size=5, num_dim_final=10, activation=torch.tanh):
        super(SimilarityEmbedding, self).__init__()
        self.layer_norm = nn.LayerNorm([num_channels, num_points])
        self.num_hidden_layers_f = num_hidden_layers_f
        self.num_hidden_layers_h = num_hidden_layers_h
        self.layers_f = ResNet(num_ifos=[3,None], layers=[2,2], kernel_size=kernel_size, context_dim=100)
        self.contraction_layer = nn.Linear(in_features=100, out_features=num_dim)
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

def train_one_epoch_se(epoch_index, data_loader, similarity_embedding, optimizer, vicreg_loss, **vicreg_kwargs):
    '''
    Training function
    Inputs: 
        epoch_index: current epoch number
        data_loader: validation data in tensor format
        similarity_embedding: ResNet to train
        optimizer: desired optimization method
        vicreg_loss: loss function
        **vicreg_kwargs: additional loss function parameters to change loss weights
    Outputs:
        last_sim_loss: final loss calculation
    '''
    running_sim_loss = 0.
    last_sim_loss = 0.
    for idx, val in enumerate(data_loader, 1):
        augmented_shift, unshifted_shift, augmented_data, unshifted_data = val      
        augmented_shift = augmented_shift.reshape((-1,)+augmented_shift.shape[2:]).to(device)
        unshifted_shift = unshifted_shift.reshape((-1,)+unshifted_shift.shape[2:]).to(device)
        augmented_data = augmented_data.reshape((-1,)+augmented_data.shape[2:]).to(device)      
        unshifted_data = unshifted_data.reshape((-1,)+unshifted_data.shape[2:]).to(device)
        embedded_values_aug, _ = similarity_embedding(augmented_data)
        embedded_values_orig, _ = similarity_embedding(unshifted_data)
        similar_embedding_loss, _repr, _cov, _std = vicreg_loss(embedded_values_aug, embedded_values_orig, **vicreg_kwargs)
        optimizer.zero_grad()
        similar_embedding_loss.backward()
        optimizer.step()
        running_sim_loss += similar_embedding_loss.item()
        n = 10
        if idx % n == 0:
            last_sim_loss = running_sim_loss / n
            print(' Avg. train loss/batch after {} batches = {:.4f}'.format(idx, last_sim_loss))
            print(f'Last {_repr.item():.2f}; {_cov.item():.2f}; {_std.item():.2f}')
            running_sim_loss = 0.
    return last_sim_loss

def val_one_epoch_se(epoch_index, data_loader, similarity_embedding, vicreg_loss, **vicreg_kwargs):
    '''
    Validation training function
    Inputs: 
        epoch_index: current epoch number
        data_loader: validation data in tensor format
        similarity_embedding: ResNet to train
        vicreg_loss: loss function
        **vicreg_kwargs: additional loss function parameters to change loss weights
    Outputs:
        last_sim_loss: final loss calculation
    '''
    running_sim_loss = 0.
    last_sim_loss = 0.
    for idx, val in enumerate(data_loader, 1):
        augmented_shift, unshifted_shift, augmented_data, unshifted_data = val
        augmented_shift = augmented_shift.reshape((-1,)+augmented_shift.shape[2:]).to(device)
        unshifted_shift = unshifted_shift.reshape((-1,)+unshifted_shift.shape[2:]).to(device)
        augmented_data = augmented_data.reshape((-1,)+augmented_data.shape[2:]).to(device)
        unshifted_data = unshifted_data.reshape((-1,)+unshifted_data.shape[2:]).to(device)
        embedded_values_aug, _ = similarity_embedding(augmented_data)
        embedded_values_orig, _ = similarity_embedding(unshifted_data)
        similar_embedding_loss, _repr, _cov, _std = vicreg_loss(embedded_values_aug, embedded_values_orig, **vicreg_kwargs)
        running_sim_loss += similar_embedding_loss.item()
        n = 1
        if idx % n == 0:
            last_sim_loss = running_sim_loss / n
            running_sim_loss = 0.
    return last_sim_loss

def train_se(config, data):
    '''
    Trains the similarity embedding according to the configured parameters
    Inputs:
        config: a dictionary of values that are supplied to tune the similarity embedding
        data: the tensors containing the training and validation set of the data
    Outputs:
        None
    '''
    # define similarity embedding and move to gpu
    print('starting training')
    similarity_embedding = SimilarityEmbedding(config["num_dim"], config["num_hidden_layers_f"], config["num_hidden_layers_h"], config["num_blocks"], config["kernel_size"], config['num_dim_final']).to(device)
    print('sim embedding defined')
    # define optimizer
    optimizer = optim.Adam(similarity_embedding.parameters(), lr=config['lr'])
    print('optimiser defined')
    #  define loss function
    vicreg_loss = VICRegLoss()
    print('loss defined')
    # sets learning rate steps
    scheduler_1 = optim.lr_scheduler.ConstantLR(optimizer, total_iters=5) #constant lr
    scheduler_2 = optim.lr_scheduler.OneCycleLR(optimizer, total_steps=20, max_lr=2e-3) #one cycle - increase and then decrease
    scheduler_3 = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler_1, scheduler_2, scheduler_3], milestones=[5, 15])
    print('scheduler defined')
    num_batches_sample = len(data)
    train_set_size = int(0.85 * num_batches_sample)
    val_set_size = num_batches_sample - train_set_size
    train_data, val_data = torch.utils.data.random_split(data, [train_set_size, val_set_size])
    train_data_loader = DataLoader(train_data, int(config["batch_size"]), shuffle=True, num_workers=4)
    val_data_loader = DataLoader(val_data, int(config["batch_size"]), shuffle=True, num_workers=4)
    print('dataset loaded')
    # check working directory
    current_working_directory = os.getcwd()
    print(current_working_directory)
    # load checkpoint (after first loop has run)
    loaded_checkpoint = session.get_checkpoint()
    if loaded_checkpoint:
        with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
            chkpath = os.path.join(loaded_checkpoint_dir, 'checkpoint.pt')
            print(chkpath)
            # load the model, optimizer, and identifier key from checkpoint
            model_state, optimizer_state, tracer_state = torch.load(os.path.join(loaded_checkpoint_dir, "checkpoint.pt"))
        similarity_embedding.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
        print('loaded checkpoint.pt')
    # train
    EPOCHS = 50
    epoch_number = 0
    for i in range(EPOCHS):
        wt_repr, wt_cov, wt_std = (1, 1, 1)
        similarity_embedding.train(True)
        avg_train_loss = train_one_epoch_se(epoch_number, train_data_loader, similarity_embedding, optimizer, vicreg_loss, wt_repr=wt_repr, wt_cov=wt_cov, wt_std=wt_std)
        # no gradient tracking, for validation
        similarity_embedding.train(False)
        similarity_embedding.eval()
        avg_val_loss = val_one_epoch_se(epoch_number, val_data_loader, similarity_embedding, vicreg_loss, wt_repr=wt_repr, wt_cov=wt_cov, wt_std=wt_std)
        print(f"Train/Val Sim Loss after epoch {epoch_number:} {avg_train_loss:.4f}/{avg_val_loss:.4f}".format(epoch_number, avg_train_loss, avg_val_loss))
        epoch_number += 1
        scheduler.step()
        # save a checkpoint
        os.makedirs("my_model", exist_ok=True)
        tracer = ['mmd', avg_train_loss, avg_val_loss, current_working_directory]
        torch.save((similarity_embedding.state_dict(), optimizer.state_dict(), tracer), "my_model/checkpoint.pt")
        checkpoint = Checkpoint.from_directory("my_model")
        session.report({"avg_train_loss":avg_train_loss, "avg_val_loss":avg_val_loss}, checkpoint=checkpoint)
        current_directory = os.getcwd()
        print(current_directory)
    print("Finished Training")

def test_best_model(best_result):
    # load the best similarity embedding
    best_trained_model = SimilarityEmbedding(best_result.config["num_dim"], best_result.config["num_hidden_layers_f"], best_result.config["num_hidden_layers_h"], best_result.config["num_blocks"], best_result.config["kernel_size"], best_result.config['num_dim_final'])
    # put the model on the gpu
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    best_trained_model.to(device)
    # pull the best result checkpoint
    current_directory = os.getcwd()
    print(current_directory)
    checkpoint_path = os.path.join(best_result.checkpoint.to_directory(), "checkpoint.pt")
    print(checkpoint_path)
    # load the model weights and save best model weights to a separate directory
    model_state, optimizer_state, tracer_state = torch.load(checkpoint_path)
    # set tracer to confirm successful save
    assert(tracer_state[0] == 'mmd')
    print(tracer_state)
    best_trained_model.load_state_dict(model_state)
    # save model
    SAVEPATH = '/nobackup/users/mmdesai/bestembeddingweights/similarity-embedding-weights.pth'
    torch.save(model_state, SAVEPATH)
    print(best_trained_model.eval())

def main(num_samples=10, max_num_epochs=50, gpus_per_trial=1):
    '''
    Overhead function for hyperparamter tuning
    INPUTS:
        num_samples: number of models with a combination of tunable parameters
        max_num_epochs: maximum number of epochs to train the model for
        gpus_per_trial: number of GPUs available 
    OUTPUTS:
        NONE, will print best trial results
    '''
    # define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    # dictionary of hyperparamter tuning choices
    config = {
        "num_dim": tune.choice([3, 4, 5, 7, 8]),
        "num_hidden_layers_f": tune.choice([1]),
        "num_hidden_layers_h": tune.choice([1, 2, 3]),
        "num_blocks": tune.choice([3, 4, 5, 6]),
        "kernel_size": tune.choice([5, 7, 9, 13]),
        "num_dim_final":tune.choice([2, 5, 10]),
        "lr": tune.loguniform(1e-6, 1e-4),
        "batch_size": tune.choice([25, 50, 75]),
    }
    # scheduler of the tuner
    tune_scheduler = ASHAScheduler(
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    # load the data
    data_shifted1 = torch.load('/nobackup/users/mmdesai/updated_tensors/data_shifted_paper4.pt')
    data_unshifted1 = torch.load('/nobackup/users/mmdesai/updated_tensors/data_unshifted_paper4.pt')
    param_shifted1 = torch.load('/nobackup/users/mmdesai/updated_tensors/param_shifted_paper4.pt')
    param_unshifted1 = torch.load('/nobackup/users/mmdesai/updated_tensors/param_unshifted_paper4.pt')
    data_shifted2 = torch.load('/nobackup/users/mmdesai/updated_tensors/data_shifted_paper5.pt')
    data_unshifted2 = torch.load('/nobackup/users/mmdesai/updated_tensors/data_unshifted_paper5.pt')
    param_shifted2 = torch.load('/nobackup/users/mmdesai/updated_tensors/param_shifted_paper5.pt')
    param_unshifted2 = torch.load('/nobackup/users/mmdesai/updated_tensors/param_unshifted_paper5.pt')
    data_shifted3 = torch.load('/nobackup/users/mmdesai/updated_tensors/data_shifted_paper6.pt')
    data_unshifted3 = torch.load('/nobackup/users/mmdesai/updated_tensors/data_unshifted_paper6.pt')
    param_shifted3 = torch.load('/nobackup/users/mmdesai/updated_tensors/param_shifted_paper6.pt')
    param_unshifted3 = torch.load('/nobackup/users/mmdesai/updated_tensors/param_unshifted_paper6.pt')
    data_shifted = torch.stack(data_shifted1 + data_shifted2 + data_shifted3)
    data_unshifted = torch.stack(data_unshifted1 + data_unshifted2 + data_unshifted3)
    param_shifted = torch.stack(param_shifted1 + param_shifted2 + param_shifted3)
    param_unshifted = torch.stack(param_unshifted1 + param_unshifted2 + param_unshifted3)
    num_batches = len(data_shifted)
    dataset = Paper_data(data_shifted, data_unshifted, param_shifted, param_unshifted, num_batches)
    model_set_size = int(0.9 * num_batches)
    test_set_size = num_batches - model_set_size
    model_data, test_data = torch.utils.data.random_split(dataset, [model_set_size, test_set_size])
    test_data_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=2)
    # tuner
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train_se, data=model_data),
            resources={"cpu": 5, "gpu": gpus_per_trial}
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
    # verify the best result and store tuned weights
    test_best_model(best_result)

main(num_samples=100, max_num_epochs=100, gpus_per_trial=1)










