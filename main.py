#!/usr/bin/python

from datetime import datetime
from argparse import ArgumentParser


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
    logdir:str = "runs", 
    configs:int = 5, 
    architecture:str = "GIN"):


    if architecture == "gin":
        model_class = baselines.GIN_Baseline
    if architecture == "sinkhorn":
        model_class = baselines.Sinkhorn_Baseline

    data_module = data.DataModule("tox21_original", split_mode = "predefined")
    
    training.search_configs(
        model_class, 
        data_module, 
        search_grid, 
        randomly_try_n = configs, 
        logdir = logdir
        )



if __name__ == '__main__':

    now = datetime.now().strftime("%d-%m-%Y_%H-%M-%S") # current date and time


    parser = ArgumentParser()
    parser.add_argument("-l", "--logdir", help="Directories where logs are stored", default=f"{now}")
    parser.add_argument("-c", "--configs", help="Number of configs to try", default=5)
    parser.add_argument("-a", "--architecture", help="The architecture of choice", default="GIN")

    args = parser.parse_args()

    logdir = "runs/" + str(args.logdir)
    configs = int(args.configs)
    architecture = str(args.architecture).lower()

    main(logdir, configs, architecture)