
"""
Utility classes for datasets which we test/train on
"""

__author__ = "Alexander Krauck"
__email__ = "alexander.krauck@gmail.com"
__date__ = "21-08-2021"

from torch_geometric.datasets import MoleculeNet
from torch_geometric.data import DataLoader

import torch

__names = ["tox21"]

class DataModule():
    """"""

    def __init__(self, data: str, root_dir: str, split_mode: str = "fixed", test_ratio: float = 0.2, batch_size: int = 64):
        """"""
        
        assert data in __names

        if data == "tox21":
            dataset = MoleculeNet(root="data/", name="Tox21")
            self.dataset_size = len(dataset)
            self.num_classes = dataset.num_classes

            torch.manual_seed(1337)
            dataset = dataset.shuffle()

            test_size = int(self.dataset_size * test_ratio)

            #test and validation set size should be equal
            test_dataset = dataset[:test_size]
            val_dataset = dataset[test_size:test_size * 2]
            train_dataset = dataset[test_size * 2:]
    
            self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            self.test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
            self.val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)





