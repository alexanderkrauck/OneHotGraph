
"""
Utility classes/functions for training models
"""

__author__ = "Alexander Krauck"
__email__ = "alexander.krauck@gmail.com"
__date__ = "21-08-2021"

import itertools
from time import time
from typing import List
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
import tqdm
import copy

from os.path import join
from os import listdir
import os

from torch.utils.tensorboard import SummaryWriter


def train(
    model: torch.nn.Module, 
    optimizer: torch.optim.Optimizer, 
    loader: torch.utils.data.DataLoader, 
    n_classes: int, 
    epoch: int, 
    logger: SummaryWriter = None, 
    dataset_class_names: List[str] = None, 
    device: str = "cpu", 
    use_tqdm: bool = False,
    **kwargs):

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
    for minibatch in iterate:

        data = minibatch[0]
        n_sample_nodes = minibatch[1].to(device)
        adjs = [adj.to(device) for adj in minibatch[2]]
        xs = [x.to(device) for x in minibatch[3]]
        x, edge_index, batch = data.x.float().to(device), data.edge_index.to(device), data.batch.to(device)
        y = data.y.to(device)

        out = model(x, edge_index, batch, n_sample_nodes = n_sample_nodes, adjs = adjs, xs = xs)

        is_not_nan = ~y.isnan()
        y = torch.nan_to_num(y, 0.5)

        try:
            loss = (F.binary_cross_entropy(out, y, reduction="none") * is_not_nan).mean() #Same as is the DeepTox paper (ignore nan preds)
        except RuntimeError:
            print(y, out)

        optimizer.zero_grad() 
        loss.backward()
        optimizer.step()

        if logger: logger.add_scalar(f"BCR_MultiTask", loss.detach().cpu().numpy(), global_step=loader.batch_size * ( n_minibatches * (epoch - 1) + batch_nr + 1))

        for i in range(n_classes):
            y = data.y[:,i]
            is_not_nan = ~y.isnan()

            indices = y[is_not_nan].long().detach().cpu().numpy()
            rs = out[is_not_nan].detach().cpu().numpy()
            probs = rs[:, i]
        
            indices_list[i].append(indices)
            probs_list[i].append(probs)

        batch_nr += 1
    
    scores = []
    for i, indices, probs in zip(range(n_classes), indices_list, probs_list):
        indices = np.concatenate(indices)
        probs = np.concatenate(probs)
        scores.append(roc_auc_score(indices, probs))
        if logger: logger.add_scalar(f"AUC-ROC/train/{dataset_class_names[i]}", scores[-1], global_step=epoch)

    if logger: logger.add_scalar(f"AUC-ROC/train/Mean_AUC_ROC", np.mean(scores), global_step=epoch)


        

def test(
    model: torch.nn.Module, 
    loader: torch.optim.Optimizer, 
    n_classes: int, 
    epoch: int, 
    logger: SummaryWriter = None, 
    dataset_class_names: List[str] = None, 
    run_type: str = "test", 
    device: str = "cpu", 
    use_tqdm: bool = False, 
    **kwargs):

    model.eval()

    indices_list = [[] for d in range(n_classes)]
    probs_list = [[] for d in range(n_classes)]

    if dataset_class_names is None:
        dataset_class_names = range(n_classes)

    if use_tqdm:
        iterate = tqdm.tqdm(loader)
    else:
        iterate = loader

    for minibatch in iterate:  # Iterate in batches over the training/test dataset.

        data = minibatch[0]
        n_sample_nodes = minibatch[1].to(device)
        adjs = [adj.to(device) for adj in minibatch[2]]
        xs = [x.to(device) for x in minibatch[3]]

        x, edge_index, batch = data.x.float().to(device), data.edge_index.to(device), data.batch.to(device)

        out = model(x, edge_index, batch, n_sample_nodes = n_sample_nodes, adjs = adjs, xs = xs)
        for i in range(n_classes):
            y = data.y[:,i]
            is_not_nan = ~y.isnan()

            indices = y[is_not_nan].long().detach().cpu().numpy()
            rs = out[is_not_nan].detach().cpu().numpy()
            probs = rs[:, i]
        
            #print(indices.shape, probs1.shape, rs.shape, np.ones_like(indices).shape)
            indices_list[i].append(indices)
            probs_list[i].append(probs)
        
    metric_dict = {}
    for i, indices, probs in zip(range(n_classes), indices_list, probs_list):
        indices = np.concatenate(indices)
        probs = np.concatenate(probs)
        score = roc_auc_score(indices, probs)
        metric_dict["AUC_ROC_"+dataset_class_names[i]] = score
        if logger: logger.add_scalar(f"AUC-ROC/{run_type}/{dataset_class_names[i]}", score, global_step=epoch)
    
    metric_dict["Mean_AUC_ROC"] = np.mean(list(metric_dict.values()))

    if logger: logger.add_scalar(f"AUC-ROC/{run_type}/Mean_AUC_ROC", metric_dict["Mean_AUC_ROC"], global_step=epoch)

    return metric_dict

def train_config(
    model_class,
    data_module,
    logger: SummaryWriter = None,
    lr = 1e-2,
    weight_decay = 1e-8,
    batch_size = 64,
    epochs = 100, 
    device = "cpu",
    save = "best",
    scheduler_patience = 10,
    scheduler_min_lr = 1e-6,
    scheduler_factor = 0.5,
    scheduler_cooldown = 3,
    **kwargs
    ):

    model = model_class(
        data_module = data_module,
        logger = logger,
        **kwargs
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer = optimizer, 
        mode = "max", 
        factor = scheduler_factor, 
        patience=scheduler_patience, 
        cooldown=scheduler_cooldown, 
        min_lr=scheduler_min_lr
        )

    train_loader = data_module.make_train_loader(batch_size = batch_size)
    val_loader = data_module.make_val_loader()
    test_loader = data_module.make_test_loader()

    test(model, train_loader, data_module.num_classes, 0, logger, data_module.class_names, run_type="train", device = device, **kwargs)
    best_metric_dict = test(model, val_loader, data_module.num_classes, 0, logger, data_module.class_names, run_type="validation", device = device, **kwargs)
    model.epoch_log(epoch = 0)
    best_epoch_dict = {}

    best_test_metric_dict = test(model, val_loader, data_module.num_classes, 0, logger, data_module.class_names, run_type="validation", device = device, **kwargs)

    for epoch in range(1, epochs + 1):
    
        train(model, optimizer, train_loader, data_module.num_classes, epoch, logger, data_module.class_names, device = device, **kwargs)
        metric_dict = test(model, val_loader, data_module.num_classes, epoch, logger, data_module.class_names, run_type="validation", device = device, **kwargs)
        model.epoch_log(epoch = epoch)
        logger.flush()
            

        better_in = []
        for metric in best_metric_dict:
            if metric_dict[metric] > best_metric_dict[metric]:
                better_in.append(metric)
                best_metric_dict[metric] = metric_dict[metric]
                best_epoch_dict[metric] = epoch

        #Evaluate on test-set to obtain the true unbiased estimate
        if len(better_in) != 0:
            test_metric_dict = test(model, test_loader, data_module.num_classes, epoch, logger, data_module.class_names, run_type="test", device = device, **kwargs)
            test_metric_dict["VAL_Mean_AUC_ROC"] = metric_dict["Mean_AUC_ROC"]
            for metric in better_in:

                best_test_metric_dict[metric] = test_metric_dict
                if save == "best":
                    torch.save(model.state_dict(), join(logger.get_logdir(),f"best_{metric}.ckp"))#this overwrites the old
                elif save == "all":
                    torch.save(model.state_dict(), join(logger.get_logdir(),f"{metric}-{test_metric_dict[metric]}.ckp"))#this not


        lr_scheduler.step(metric_dict["Mean_AUC_ROC"])
        if lr_scheduler.num_bad_epochs == scheduler_patience and lr_scheduler._last_lr[0] <= scheduler_min_lr + lr_scheduler.eps:
            #if this is true => we are done!
            break

        


    return best_test_metric_dict, best_epoch_dict


    

def dict_product(dicts):

    return (dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))

def grid_search_configs(
    model_class,
    data_module, 
    search_grid, 
    randomly_try_n = -1, 
    logdir = "runs", 
    **kwargs):

    configurations = [config for config in dict_product(search_grid)]
    print(f"Total number of Grid-Search configurations: {len(configurations)}")
    if os.path.exists(logdir):
        existing_dirs = listdir(logdir)
    else:
        existing_dirs = []

    if randomly_try_n == -1:
        randomly_try_n = len(configurations)
    
    print(f"Number of configurations now being trained {randomly_try_n}")
    print("--------------------------------------------------------------------------------------------\n")
    
    tried = 0
    while randomly_try_n > tried and len(configurations) != 0:

        tried += 1

        idx = np.random.choice(len(configurations))
        config = configurations.pop(idx)
        config_str = str(config).replace("'","").replace(":", "-").replace(" ", "").replace("}", "").replace("_","").replace(",", "_").replace("{","_")
        if config_str in existing_dirs:
            continue

        print(f"Training config {config_str} ... ", end="")
        dt = time()
    

        logger = SummaryWriter(log_dir = logdir + "/" + config_str, comment = config_str)        

        metric_dict, epoch_dict = train_config(
            model_class = model_class,
            data_module = data_module,
            logger = logger,
            **config,
            **kwargs
            )

        inverse_dict = dict((v,k) for k,v in epoch_dict.items())
        for epoch in inverse_dict:
            metric = inverse_dict[epoch]

            copied_conf = copy.deepcopy(config)
            copied_conf["epoch"] = epoch
            curr_metric_dict = metric_dict[metric]
            new_metric_dict = {}
            for curr_metric in curr_metric_dict:
                new_metric_dict["hparam/"+curr_metric] = curr_metric_dict[curr_metric]


            if logger: logger.add_hparams(copied_conf, new_metric_dict, run_name= f"ep{epoch}")
            
        print(f"done (took {time() - dt:.2f}s)")