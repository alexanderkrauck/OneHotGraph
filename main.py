#!/usr/bin/python

from datetime import datetime
from argparse import ArgumentParser

import torch

import utils.data as data
import utils.training as training
import utils.baselines as baselines
import utils.sinkhorn_graph as sinkhorn_graph

search_grid = {
    "hidden_channels": [64, 256, 1028],
    "head_depth": [1,2,3,4],
    "base_depth": [3,5,10],
    "base_dropout": [0.5, 0.2],
    "head_dropout": [0.5, 0.2],
    "lr": [1e-2, 1e-3]
}



def main(
    name:str = "*time*",#*time* is replaced by the datetime
    logdir:str = "runs",
    configs:int = 5, 
    architecture:str = "GIN",
    device:str = "cpu"):

    #TODO: Add device check
    if torch.cuda.is_available():
        if device.isdigit():
            device_n = int(device)
            if device_n < torch.cuda.device_count():
                device = torch.cuda.device(device_n)
            else:
                device = "cpu"
            
    name = name.replace("*time*", datetime.now().strftime("%d-%m-%Y_%H-%M-%S"))


    if architecture == "gin":
        model_class = baselines.GIN_Baseline
    if architecture == "sinkhorn":
        model_class = baselines.Sinkhorn_Baseline
    if architecture == "gat":
        model_class = baselines.GAT_Baseline

    data_module = data.DataModule("tox21_original", split_mode = "predefined")
    
    training.search_configs(
        model_class, 
        data_module, 
        search_grid, 
        randomly_try_n = configs, 
        logdir = logdir + "/" + name,
        device = device
        )



if __name__ == '__main__':



    parser = ArgumentParser()
    parser.add_argument("-n", "--name", help="Name of the experiment", default=f"*time*")
    parser.add_argument("-l", "--logdir", help="Directories where logs are stored", default=f"runs")
    parser.add_argument("-c", "--configs", help="Number of configs to try", default=5)
    parser.add_argument("-a", "--architecture", help="The architecture of choice", default="GIN")
    parser.add_argument("-d", "--device", help="The device of choice", default="cpu")

    args = parser.parse_args()

    logdir = str(args.logdir)
    name = str(args.name)
    configs = int(args.configs)
    architecture = str(args.architecture).lower()
    device = str(args.device)

    main(name, logdir, configs, architecture, device)