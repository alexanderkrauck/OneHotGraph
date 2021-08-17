
"""
Utility classes/functions for training models
"""

__author__ = "Alexander Krauck"
__email__ = "alexander.krauck@gmail.com"
__date__ = "21-08-2021"

import itertools
from time import time
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
import tqdm


from torch.utils.tensorboard import SummaryWriter


def train(model, optimizer, loader, n_classes, epoch: int, logger: SummaryWriter, dataset_class_names = None, device = "cpu", use_tqdm = False):

    model.train()

    if dataset_class_names is None:
        dataset_class_names = range(n_classes)

    n_minibatches = len(loader)

    indices_list = [[] for d in range(n_classes)]
    probs_list = [[] for d in range(n_classes)]

    if use_tqdm:
        iterate = tqdm.tqdm(loader)
    else:
        iterate = loader

    batch_nr = 0
    for data in iterate:
        x, edge_index, batch = data.x.float().to(device), data.edge_index.to(device), data.batch.to(device)

        out = model(x, edge_index, batch)

        y = data.y.to(device)
        is_not_nan = ~y.isnan()
        y = torch.nan_to_num(y, 0.5)

        try:
            loss = (F.binary_cross_entropy(out, y, reduction="none") * is_not_nan).mean() #Same as is the DeepTox paper
        except RuntimeError:
            print(y, out)

        optimizer.zero_grad() 
        loss.backward()
        optimizer.step()

        logger.add_scalar(f"BCR_MultiTask", loss.detach().cpu().numpy(), global_step=n_minibatches * (epoch - 1) + batch_nr + 1)

        for i in range(n_classes):
            y = data.y[:,i]
            is_not_nan = ~y.isnan()

            indices = y[is_not_nan].long().detach().cpu().numpy()
            rs = out[is_not_nan].detach().cpu().numpy()
            probs = rs[:, i]
        
            #print(indices.shape, probs1.shape, rs.shape, np.ones_like(indices).shape)
            indices_list[i].append(indices)
            probs_list[i].append(probs)

        batch_nr += 1
    
    for i, indices, probs in zip(range(n_classes), indices_list, probs_list):
        indices = np.concatenate(indices)
        probs = np.concatenate(probs)
        logger.add_scalar(f"AUC-ROC/train/{dataset_class_names[i]}", roc_auc_score(indices, probs), global_step=epoch)

        

def test(model, loader, n_classes, epoch:int, logger: SummaryWriter, dataset_class_names = None, run_type = "test", device = "cpu", use_tqdm = False):
    model.eval()

    indices_list = [[] for d in range(n_classes)]
    probs_list = [[] for d in range(n_classes)]

    if dataset_class_names is None:
        dataset_class_names = range(n_classes)

    if use_tqdm:
        iterate = tqdm.tqdm(loader)
    else:
        iterate = loader

    for data in iterate:  # Iterate in batches over the training/test dataset.
        x, edge_index, batch = data.x.float().to(device), data.edge_index.to(device), data.batch.to(device)
        #batch containts for each x to which sample in the minibatch it belongs.

        out = model(x, edge_index, batch)

        for i in range(n_classes):
            y = data.y[:,i]
            is_not_nan = ~y.isnan()

            indices = y[is_not_nan].long().detach().cpu().numpy()
            rs = out[is_not_nan].detach().cpu().numpy()
            probs = rs[:, i]
        
            #print(indices.shape, probs1.shape, rs.shape, np.ones_like(indices).shape)
            indices_list[i].append(indices)
            probs_list[i].append(probs)
        
    for i, indices, probs in zip(range(n_classes), indices_list, probs_list):
        indices = np.concatenate(indices)
        probs = np.concatenate(probs)
        logger.add_scalar(f"AUC-ROC/{run_type}/{dataset_class_names[i]}", roc_auc_score(indices, probs), global_step=epoch)

    return 0.5, 0.5 # Derive ratio of correct predictions.

def train_config(
    model_class,
    data_module,
    logger: SummaryWriter,
    hidden_channels = 128,
    head_depth = 3, 
    base_depth = 5, 
    base_dropout = 0.5, 
    head_dropout = 0.5, 
    lr = 1e-2,
    weight_decay = 1e-8,
    batch_size = 64,
    epochs = 100, 
    device = "cpu"
    ):

    model = model_class(
        data_module = data_module,
        n_hidden_channels = hidden_channels,
        n_graph_layers = base_depth, 
        n_graph_dropout = base_dropout, 
        n_linear_layers = head_depth, 
        n_linear_dropout = head_dropout
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


    train_loader = data_module.make_train_loader(batch_size = batch_size)
    val_loader = data_module.make_val_loader()

    test(model, train_loader, data_module.num_classes, 0, logger, data_module.class_names, run_type="train", device = device)
    test(model, val_loader, data_module.num_classes, 0, logger, data_module.class_names, run_type="validation", device = device)

    for epoch in range(1, epochs + 1):
    
        train(model, optimizer, train_loader, data_module.num_classes, epoch, logger, data_module.class_names, device = device)
        test(model, val_loader, data_module.num_classes, epoch, logger, data_module.class_names, run_type="validation", device = device)

        logger.flush()

def dict_product(dicts):

    return (dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))

def search_configs(model_class, data_module, search_grid, randomly_try_n = -1, logdir = "runs", device = "cpu"):

    configurations = [config for config in dict_product(search_grid)]
    print(f"Total number of Grid-Search configurations: {len(configurations)}")

    if randomly_try_n == -1:
        do_indices = range(len(configurations))
    else:
        do_indices = np.random.choice(len(configurations), size=randomly_try_n)
    
    print(f"Number of configurations now being trained {len(do_indices)}")
    print("--------------------------------------------------------------------------------------------\n")
    
    for trial_nr, idx in enumerate(do_indices):
        
        config = configurations[idx]


        config_str = str(config).replace("'","").replace(":", "-").replace(" ", "").replace("}", "").replace("_","").replace(",", "_").replace("{","_")

        print(f"Training config {config_str} ... ", end="")
        dt = time()
    

        logger = SummaryWriter(log_dir = logdir + "/" + config_str, comment = config_str)
        #logger.add_hparams(config,  ,run_name= f"run{trial_nr}")

        train_config(
            model_class = model_class,
            data_module = data_module,
            logger = logger,
            hidden_channels = config["hidden_channels"], 
            head_depth = config["head_depth"], 
            base_depth =  config["base_depth"], 
            base_dropout =  config["base_dropout"], 
            head_dropout =  config["head_dropout"], 
            weight_decay= config["weight_decay"],
            lr =  config["lr"], 
            batch_size = config["batch_size"],
            device = device
            )
            
        print(f"done (took {time() - dt:.2f}s)")