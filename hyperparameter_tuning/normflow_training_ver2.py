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
from resnet import ResNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

num_repeats = 50
num_channels = 3
num_points = 121
in_features = num_points

class Paper_data(Dataset):
    '''
    Dataset Processing
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
        x = self.contraction_layer(x)
        representation = torch.clone(x)
        x = self.activation(self.expander_layer(x))
        for layer in self.layers_h:
            x = layer(x)
            x = self.activation(x)
        x = self.final_layer(x)
        return x, representation

class EmbeddingNet(nn.Module):
    '''
    Wrapper around the similarity embedding defined above
    '''
    def __init__(self, similarity_embedding, num_dim,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.representation_net = similarity_embedding
        self.num_dim = num_dim
        self.representation_net.load_state_dict(similarity_embedding.state_dict())
        # the expander network is unused and hence don't track gradients
        for name, param in self.representation_net.named_parameters():
            if 'expander_layer' in name or 'layers_h' in name or 'final_layer' in name:
                param.requires_grad = False
        # set freeze status of part of the conv layer of embedding_net
            elif 'layers_f' in name:
                param.requires_grad = True
            else: 
                param.requires_grad = True
        self.context_layer = nn.Sequential(
            nn.Linear(num_dim, 1000),
            nn.ReLU(),
            nn.Linear(1000, num_dim)
            )
    def forward(self, x):
        batch_size, channels, dims = x.shape
        num_dim = 8
        _, rep = self.representation_net(x)
        rep = rep.reshape(batch_size, num_dim)

        return self.context_layer(rep)

def normflow_params(similarity_embedding, num_transforms, num_blocks, hidden_features, context_features, num_dim):
    '''
    Normalizing flow strucutre
    '''
    base_dist = StandardNormal([3])
    transforms = []
    features = 3
    for _ in range(num_transforms):
        block = [MaskedAffineAutoregressiveTransform(
                features=3,  
                hidden_features=hidden_features,  
                context_features=context_features,  
                num_blocks=num_blocks,   
                activation=torch.tanh,
                use_batch_norm=False,
                use_residual_blocks=True,
                dropout_probability=0.01
            ),
            RandomPermutation(features=features)
        ]
        transforms += block
    transform = CompositeTransform(transforms)
    embedding_net = EmbeddingNet(similarity_embedding, num_dim)
    return transform, base_dist, embedding_net

def train_one_epoch(epoch_index, data_loader, flow, optimizer, flatten_dim, num_dim):
    '''
    Training function
    Inputs: 
        epoch_index: current epoch number
        data_loader: validation data in tensor format
        flow: normalizing flow to train
        optimizer: desired optimization method
        flatten_dim: dimension along which to flatten the tensor data
        num_dim: dimensionality of the similarity embedding
    Outputs:
        last_loss: final loss calculation
    '''
    running_loss = 0.
    last_loss = 0.
    for idx, val in enumerate(data_loader, 1):
        augmented_shift, unshifted_shift, augmented_data, unshifted_data = val
        augmented_shift = augmented_shift[...,0:3].to(device)
        augmented_shift = augmented_shift.flatten(0, flatten_dim).to(device)
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


def val_one_epoch(epoch_index, data_loader, flow, flatten_dim, num_dim):
    '''
    Training function
    Inputs: 
        epoch_index: current epoch number
        data_loader: validation data in tensor format
        flow: normalizingn flow to train
        flatten_dim: dimension along which to flatten the tensor data
        num_dim: dimensionality of the similarity embedding
    Outputs:
        last_sim_loss: final loss calculation
    '''
    running_loss = 0.
    last_loss = 0.
    for idx, val in enumerate(data_loader, 1):
        augmented_shift, unshifted_shift, augmented_data, unshifted_data = val
        augmented_shift = augmented_shift[...,0:3].to(device)
        augmented_shift = augmented_shift.flatten(0, flatten_dim).to(device)
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

def train_flow(config, data):
    '''
    Trains the normalizing flow according to the configured parameters
    Inputs:
        config: a dictionary of values that are supplied to tune the normalizing flow
        data: the tensors containing the training and validation set of the data
    Outputs:
        None
    '''
    print('starting training')
    # get the best similarity embedding
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOADPATH = '/nobackup/users/mmdesai/bestembeddingweights/similarity-embedding-weights.pth'
    similarity_embedding = SimilarityEmbedding(num_dim=8, num_hidden_layers_f=1, num_hidden_layers_h=2, num_blocks=4, kernel_size=13, num_dim_final=2).to(device)
    similarity_embedding.load_state_dict(torch.load(LOADPATH, map_location=device))
    similarity_embedding.eval()
    num_dim = 8
    print(similarity_embedding.eval())
    # dataset
    num_batches_sample = len(data)
    train_set_size = int(0.85 * num_batches_sample)
    val_set_size = num_batches_sample - train_set_size
    train_data, val_data = torch.utils.data.random_split(data, [train_set_size, val_set_size])
    train_data_loader = DataLoader(train_data, int(config["batch_size"]), shuffle=True, num_workers=4)
    val_data_loader = DataLoader(val_data, int(config["batch_size"]), shuffle=True, num_workers=4)
    print('dataset loaded')
    # define the flow
    #_, rep = similarity_embedding(var_data_se)  # _.shape = batch_size x 1 x 10, # rep.shape = batch_size x 1 x 2
    context_features = num_dim 
    transform, base_dist, embedding_net = normflow_params(similarity_embedding, config['num_transforms'], config['num_blocks'], config['hidden_features'], context_features=context_features, num_dim=num_dim)
    flow = Flow(transform, base_dist, embedding_net).to(device=device)
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
    EPOCHS = 200
    epoch_number = 0
    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))
        flow.train(True)
        for name, param in flow._embedding_net.named_parameters():
            param.requires_grad = True
        avg_train_loss = train_one_epoch(epoch_number, train_data_loader, flow, optimizer, 2, num_dim=num_dim)
        flow.train(False)
        avg_val_loss = val_one_epoch(epoch_number, val_data_loader, flow, 2, num_dim=num_dim)
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
    num_dim = 8
    context_features = num_dim 
    best_transform, best_base_dist, best_embedding_net = normflow_params(similarity_embedding, best_result.config['num_transforms'], best_result.config['num_blocks'], best_result.config['hidden_features'], context_features=context_features, num_dim=num_dim)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    best_flow = Flow(best_transform, best_base_dist, best_embedding_net).to(device=device)
    checkpoint_path = os.path.join(best_result.checkpoint.to_directory(), "checkpoint.pt")
    flow_state, optimizer_state = torch.load(checkpoint_path)
    best_flow.load_state_dict(flow_state)
    model_state, optimizer_state = torch.load(checkpoint_path)
    SAVEPATH = '/nobackup/users/mmdesai/bestflowweights/flow-weights.pth'
    torch.save(flow_state, SAVEPATH)
    print('saved model weights')

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    config = {
        'num_transforms': tune.choice([4, 5, 6, 7, 8, 9]),
        'num_blocks': tune.choice([5, 6, 7, 8, 9, 10]),
        'hidden_features': tune.choice([60, 70, 80, 90, 100]),
        "lr": tune.loguniform(1e-6, 1e-4),
        "batch_size": tune.choice([25, 50, 100]),
    }
    # dataset
    data_shifted_flow1 = torch.load('/nobackup/users/mmdesai/updated_tensors/data_shifted_flow4.pt')
    data_unshifted_flow1 = torch.load('/nobackup/users/mmdesai/updated_tensors/data_unshifted_flow4.pt')
    param_shifted_flow1 = torch.load('/nobackup/users/mmdesai/updated_tensors/param_shifted_flow4.pt')
    param_unshifted_flow1 = torch.load('/nobackup/users/mmdesai/updated_tensors/param_unshifted_flow4.pt')
    data_shifted_flow2 = torch.load('/nobackup/users/mmdesai/updated_tensors/data_shifted_flow4.pt')
    data_unshifted_flow2 = torch.load('/nobackup/users/mmdesai/updated_tensors/data_unshifted_flow5.pt')
    param_shifted_flow2 = torch.load('/nobackup/users/mmdesai/updated_tensors/param_shifted_flow5.pt')
    param_unshifted_flow2 = torch.load('/nobackup/users/mmdesai/updated_tensors/param_unshifted_flow5.pt')
    data_shifted_flow3 = torch.load('/nobackup/users/mmdesai/updated_tensors/data_shifted_flow6.pt')
    data_unshifted_flow3 = torch.load('/nobackup/users/mmdesai/updated_tensors/data_unshifted_flow6.pt')
    param_shifted_flow3 = torch.load('/nobackup/users/mmdesai/updated_tensors/param_shifted_flow6.pt')
    param_unshifted_flow3 = torch.load('/nobackup/users/mmdesai/updated_tensors/param_unshifted_flow6.pt')
    data_shifted = torch.stack(data_shifted_flow1 + data_shifted_flow2 + data_shifted_flow3)
    data_unshifted = torch.stack(data_unshifted_flow1 + data_unshifted_flow2 + data_unshifted_flow3)
    param_shifted = torch.stack(param_shifted_flow1 + param_shifted_flow2 + param_shifted_flow3)
    param_unshifted = torch.stack(param_unshifted_flow1 + param_unshifted_flow2 + param_unshifted_flow3)
    print(data_shifted[0].shape, param_shifted[0].shape)
    print('data loaded')
    num_batches = len(data_shifted)
    dataset = Paper_data(data_shifted, data_unshifted, param_shifted, param_unshifted, num_batches)
    model_set_size = int(0.9 * num_batches)
    test_set_size = num_batches - model_set_size
    model_data, test_data = torch.utils.data.random_split(dataset, [model_set_size, test_set_size])
    test_data_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=2)
    tune_scheduler = ASHAScheduler(
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train_flow, data=model_data),
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

main(num_samples=100, max_num_epochs=100, gpus_per_trial=1)



