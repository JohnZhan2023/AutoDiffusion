"""
Script to learn MDP model from data for offline policy optimization
"""

import argparse
import os
import pickle

from matplotlib import pyplot as plt
import numpy as np

from dataloader import load_data4AD
from WorldModel import WorldModel
import wandb
import torch
import yaml

def load_config(path):
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def construct_parser():
    parser = argparse.ArgumentParser(description='Training Dynamic Functions.')
    #parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training (-1 means no distributed training)")
    parser.add_argument("--config_path", default="/cephfs/zhanjh/DiffusionForcing/StateTransformer/diffusion4trajectory/conf/ccil.yaml",help="Path to config file")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--index_path", default="/cephfs/shared/nuplan/online_s6/index")
    parser.add_argument("--data_path", default="/cephfs/shared/nuplan/online_s6")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    return parser


def plot_loss(train_loss, fn, xbar=None, title='Train Loss'):
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_title("Dynamics Model Loss")
    ax.set_ylabel(title)
    ax.set_xlabel("Epoch")
    if xbar:
        ax.axhline(xbar, linestyle="--", color="black")
    ax.plot(train_loss)
    fig.savefig(fn)


def save_loss(train_loss, folder_name, prediction_error, model_name=None, eval_loss=None):
    model_name = f"{model_name}_" if model_name else ""
    for loss_name, losses in train_loss.items():
      fn_prefix = os.path.join(folder_name, f'{model_name}train_{loss_name}')
      plot_loss(losses, fn_prefix + '.png', title=loss_name)
      with open(fn_prefix+'.txt', 'w') as f:
        _l = np.array2string(np.array(losses), formatter={'float_kind':lambda x: "%.6f\n" % x})
        f.write(_l)
    with open(os.path.join(folder_name, f'{model_name}statistics.txt'), 'w') as f:
      f.write(f'Avg Prediction Error (unnormalized) {prediction_error:.16f}')

def plot_lipschitz_dist(lipschitz_coeff, folder_name, model_name=None, lipschitz_constraint=None):
    model_name = f"{model_name}_" if model_name else ""
    fig = plt.figure()
    ax = fig.add_subplot()
    fig.suptitle("Local Lipschitz Coefficient")
    ax.hist(lipschitz_coeff, density=True, bins=50)
    if lipschitz_constraint:
        ax.axvline(lipschitz_constraint, linestyle="--", color="black")
    path = os.path.join(folder_name, f"{model_name}train_local_lipschitz.png")
    fig.savefig(path)

def main():
    wandb.init(project='ccil for self driving', entity='zhanjiahao384')
    parser = construct_parser()
    args = parser.parse_args()
    

    import random
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    # Load Data
    train_dataloader=load_data4AD(args.index_path,args.data_path,"train",batch_size=args.batch_size)
    
    # Load Config
    config = load_config(args.config_path)

    dynamics = WorldModel(config)

    # we should move the model to the GPU after the optimizer is created
    print("cuda available:",torch.cuda.is_available())
    dynamics.to("cuda:0")
    # Fit Dynamics Model
    # learn forward dynamics
    
    train_loss = dynamics.fit_AD_dynamics(train_dataloader,50)
    
    # Report Validation Loss
    del train_dataloader
    test_dataloader=load_data4AD(args.index_path,args.data_path,"test",batch_size=args.batch_size)
    prediction_error = dynamics.eval_prediction_error(test_dataloader)
    print("Prediction Error: ", prediction_error)
if __name__ == "__main__":
    
    main()
