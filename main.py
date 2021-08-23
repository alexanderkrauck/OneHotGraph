#!/usr/bin/python


import yaml

from datetime import datetime
from argparse import ArgumentParser

import torch

import utils.data as data
import utils.training as training
import utils.baselines as baselines
import utils.sinkhorn_graph as sinkhorn_graph

default_search_grid = {
    "hidden_channels": [64, 256, 1028],
    "head_depth": [1,2,3,4],
    "base_depth": [3,5,10],
    "base_dropout": [0.5, 0.2],
    "head_dropout": [0.5, 0.2],
    "lr": [1e-2, 1e-3],
    "weight_decay": [0, 1e-10, 1e-8, 1e-5],
    "batch_size": [256, 64, 16]
}

def main(
    name:str = "*time*",#*time* is replaced by the datetime
    logdir:str = "runs",
    configs:int = 5, 
    architecture:str = "GIN",
    device:str = "cpu",
    epochs: int = 100,
    save: str = "best",
    workers: int = 2,
    yaml_file: str = ""):

    #TODO: Add device check
    if torch.cuda.is_available():
        if device.isdigit():
            device_n = int(device)
            if device_n < torch.cuda.device_count():
                device = "cuda:" + device
            else:
                device = "cpu"
            
    name = name.replace("*time*", datetime.now().strftime("%d-%m-%Y_%H-%M-%S"))

    data_module = data.DataModule("tox21_original", split_mode = "predefined", workers = workers)

    n_architectures = len(architecture.split(","))

    if yaml_file == "":
        search_grid = default_search_grid
    else:
        with open(yaml_file, 'r') as file:
            search_grid = yaml.safe_load(file)

    for a in architecture.split(","):
        if a == "gin":
            model_class = baselines.GIN_Baseline
        elif a == "sinkhorn":
            model_class = baselines.Sinkhorn_Baseline
        elif a == "gat":
            model_class = baselines.GAT_Baseline
        elif a == "gingat":
            model_class = baselines.GINGAT_Baseline
        elif a == "onehotgraph" or a == "onehot" or a == "ohg":
            model_class = baselines.OneHotGraph_Baseline
        else:
            print(f"Model Class {a} is unknown... skipping")
            continue

        if n_architectures > 1:
            ldir = logdir + "/" + name + "/" + a
        else:
            ldir = logdir + "/" + name
        
        training.search_configs(
            model_class, 
            data_module, 
            search_grid, 
            randomly_try_n = configs, 
            logdir = ldir,
            device = device,
            epochs = epochs,
            save = save
            )



if __name__ == '__main__':


    parser = ArgumentParser()
    parser.add_argument("-n", "--name", help="Name of the experiment", default=f"*time*")
    parser.add_argument("-l", "--logdir", help="Directories where logs are stored", default=f"runs")
    parser.add_argument("-c", "--configs", help="Number of configs to try", default=5)
    parser.add_argument("-a", "--architecture", help="The architecture of choice", default="ohg")
    parser.add_argument("-d", "--device", help="The device of choice", default="cpu")
    parser.add_argument("-e", "--epochs", help="The number of epochs to run for each config", default=100)
    parser.add_argument("-s", "--save", help="The save mode", default="best")
    parser.add_argument("-w", "--workers", help="The number of workers the dataloaders use", default=2)
    parser.add_argument("-y", "--yaml", help="The yaml file with the search grid", default = "")



    args = parser.parse_args()

    logdir = str(args.logdir)
    name = str(args.name)
    configs = int(args.configs)
    architecture = str(args.architecture).lower()
    device = str(args.device)
    epochs = int(args.epochs)
    save = str(args.save)
    workers = int(args.workers)
    yaml_file = str(args.yaml)

    main(name, logdir, configs, architecture, device, epochs, save, workers, yaml_file)