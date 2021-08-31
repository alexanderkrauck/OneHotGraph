#!/usr/bin/python


import yaml

from datetime import datetime
from argparse import ArgumentParser

import torch

import utils.data as data
import utils.training as training
import utils.baselines as baselines

default_search_grid = {
    "hidden_channels": [64, 256, 1024],
    "head_depth": [1, 2, 3, 4],
    "base_depth": [3, 5, 10],
    "base_dropout": [0.5, 0.2],
    "head_dropout": [0.5, 0.2],
    "lr": [1e-2, 1e-3],
    "weight_decay": [0, 1e-10, 1e-8, 1e-5],
    "batch_size": [256, 64, 16],
}


def main(
    name: str = "*time*",  # *time* is replaced by the datetime
    logdir: str = "runs",
    configs: int = 5,
    architecture: str = "GIN",
    device: str = "cpu",
    epochs: int = 100,
    save: str = "best",
    workers: int = 2,
    yaml_file: str = "",
    use_tqdm: bool = False,
    always_test: bool = False,
):

    if torch.cuda.is_available():
        if device.isdigit():
            device_n = int(device)
            if device_n < torch.cuda.device_count():
                device = "cuda:" + device
            else:
                device = "cpu"

    name = name.replace("*time*", datetime.now().strftime("%d-%m-%Y_%H-%M-%S"))

    data_module = data.DataModule(
        "tox21_original", split_mode="predefined", workers=workers
    )

    n_architectures = len(architecture.split(","))

    if yaml_file == "":
        search_grid = default_search_grid
    else:
        with open(yaml_file, "r") as file:
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
        elif a == "attentiononehotgraph" or a == "attentiononehot" or a == "aohg":
            model_class = baselines.AttentionOneHotGraph_Baseline
        elif a == "isomorphismonehotgraph" or a == "isomorphismonehot" or a == "iohg":
            model_class = baselines.IsomorphismOneHotGraph_Baseline
        elif a == "gcn":
            model_class = baselines.GCN_Baseline
        else:
            print(f"Model Class {a} is unknown... skipping")
            continue

        if n_architectures > 1:
            ldir = logdir + "/" + name + "/" + a
        else:
            ldir = logdir + "/" + name

        training.grid_search_configs(
            model_class,
            data_module,
            search_grid,
            randomly_try_n=configs,
            logdir=ldir,
            device=device,
            epochs=epochs,
            save=save,
            use_tqdm=use_tqdm,
            always_test=always_test,
        )


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        "-n", "--name", help="Name of the experiment", default=f"*time*", type=str
    )
    parser.add_argument(
        "-l",
        "--logdir",
        help="Directories where logs are stored",
        default=f"runs",
        type=str,
    )
    parser.add_argument(
        "-c", "--configs", help="Number of configs to try", default=-1, type=int
    )
    parser.add_argument(
        "-a",
        "--architecture",
        help="The architecture of choice",
        default="gcn",
        type=str,
    )
    parser.add_argument(
        "-d", "--device", help="The device of choice", default="cuda", type=str
    )
    parser.add_argument(
        "-e",
        "--epochs",
        help="The number of epochs to run for each config",
        default=200,
        type=int,
    )
    parser.add_argument("-s", "--save", help="The save mode", default="best")
    parser.add_argument(
        "-w",
        "--workers",
        help="The number of workers the dataloaders use",
        default=4,
        type=int,
    )
    parser.add_argument(
        "-y",
        "--yaml",
        help="The yaml file with the search grid",
        default="grids/grid_manual.yml",
        type=str,
    )
    parser.add_argument(
        "-t", "--use_tqdm", help="Whether to use tqdm or not", default="True", type=bool
    )
    parser.add_argument(
        "-t",
        "--use_tqdm",
        help="Whether to use tqdm or not",
        action="store_true",
        type=bool,
    )
    parser.add_argument(
        "--always_test",
        help="If each epoch the testset should be evaluated regardingless of the validation score",
        action="store_true",
        type=bool,
    )

    args = parser.parse_args()

    logdir = args.logdir
    name = args.name
    configs = args.configs
    architecture = args.architecture.lower()
    device = args.device
    epochs = args.epochs
    save = args.save
    workers = args.workers
    yaml_file = args.yaml
    use_tqdm = args.use_tqdm
    always_test = args.always_test

    main(
        name,
        logdir,
        configs,
        architecture,
        device,
        epochs,
        save,
        workers,
        yaml_file,
        use_tqdm,
        always_test,
    )

