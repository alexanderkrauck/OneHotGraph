"""
Utility classes for datasets which we test/train on
"""

__author__ = "Alexander Krauck"
__email__ = "alexander.krauck@gmail.com"
__date__ = "21-08-2021"



import torch_geometric
from torch_geometric.data import Data
from torch_geometric.datasets import MoleculeNet, TUDataset
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.data.dataloader import Collater

import torch

import pandas as pd
import numpy as np

import os
import re

from rdkit import Chem


_names = [
    "tox21",  # The tox21 dataset which is provided in the torch_geometric.datasets.MoleculeNet
    "tox21_original",  # The tox21 dataset how it was originally and avaiable at
    "proteins",
    "nci1"
]

x_map = {
    "atomic_num": list(range(0, 119)),
    "chirality": [
        "CHI_UNSPECIFIED",
        "CHI_TETRAHEDRAL_CW",
        "CHI_TETRAHEDRAL_CCW",
        "CHI_OTHER",
    ],
    "degree": list(range(0, 11)),
    "formal_charge": list(range(-5, 7)),
    "num_hs": list(range(0, 9)),
    "num_radical_electrons": list(range(0, 5)),
    "hybridization": [
        "UNSPECIFIED",
        "S",
        "SP",
        "SP2",
        "SP3",
        "SP3D",
        "SP3D2",
        "OTHER",
    ],
    "is_aromatic": [False, True],
    "is_in_ring": [False, True],
}

e_map = {
    "bond_type": ["misc", "SINGLE", "DOUBLE", "TRIPLE", "AROMATIC",],
    "stereo": [
        "STEREONONE",
        "STEREOZ",
        "STEREOE",
        "STEREOCIS",
        "STEREOTRANS",
        "STEREOANY",
    ],
    "is_conjugated": [False, True],
}

#Parts from the class below are taken from the MoleculeNet class of Pytorch Geometric
class ExtendedMoleculeNet(MoleculeNet):

    # Format: name: [display_name, url_name, csv_name, smiles_idx, y_idx]
    names = {
        "esol": ["ESOL", "delaney-processed.csv", "delaney-processed", -1, -2],
        "freesolv": ["FreeSolv", "SAMPL.csv", "SAMPL", 1, 2],
        "lipo": ["Lipophilicity", "Lipophilicity.csv", "Lipophilicity", 2, 1],
        "pcba": ["PCBA", "pcba.csv.gz", "pcba", -1, slice(0, 128)],
        "muv": ["MUV", "muv.csv.gz", "muv", -1, slice(0, 17)],
        "hiv": ["HIV", "HIV.csv", "HIV", 0, -1],
        "bace": ["BACE", "bace.csv", "bace", 0, 2],
        "bbbp": ["BBPB", "BBBP.csv", "BBBP", -1, -2],
        "tox21": ["Tox21", "tox21.csv.gz", "tox21", -1, slice(0, 12)],
        "toxcast": ["ToxCast", "toxcast_data.csv.gz", "toxcast_data", 0, slice(1, 618)],
        "sider": ["SIDER", "sider.csv.gz", "sider", 0, slice(1, 28)],
        "clintox": ["ClinTox", "clintox.csv.gz", "clintox", 0, slice(1, 3)],
        "tox21_original": [
            "Tox21_original",
            "tox21_original.csv.gz",
            "tox21_original.sdf",
            "infofile.csv",
            -1,
            slice(0, 12),
        ],
    }

    def download(self):
        assert self.name != "tox21_original"

        super(ExtendedMoleculeNet, self).download()

    @property
    def raw_file_names(self):
        if self.name == "tox21_original":
            return f"{self.names[self.name][2]}", f"{self.names[self.name][3]}"
        else:
            return f"{self.names[self.name][2]}.csv"

    def process(self):
        if self.name == "tox21_original":
            dataset = Chem.SDMolSupplier(self.raw_paths[0])
            info_file = pd.read_csv(self.raw_paths[1], sep=",", header=0)
            ids = info_file["sdftitle"].to_numpy()
            set = info_file["set"].to_numpy()
            targets = (info_file.to_numpy()[:, -12:]).astype(np.float32)

        else:
            with open(self.raw_paths[0], "r") as f:
                dataset = f.read().split("\n")[1:-1]
                dataset = [x for x in dataset if len(x) > 0]  # Filter empty lines.

        data_list = []
        for idx, line in enumerate(dataset):
            if self.name != "tox21_original":
                line = re.sub(r"\".*\"", "", line)  # Replace ".*" strings.
                line = line.split(",")

                smiles = line[self.names[self.name][3]]
                ys = line[self.names[self.name][4]]
                ys = ys if isinstance(ys, list) else [ys]

                ys = [float(y) if len(y) > 0 else float("NaN") for y in ys]
                y = torch.tensor(ys, dtype=torch.float).view(1, -1)

                mol = Chem.MolFromSmiles(smiles)
            else:
                mol = line
                if mol is None:
                    print(f"Skipped SDF at index {idx}. (Not readable)")
                    continue
                assert mol.GetProp("_Name") == ids[idx]
                y = torch.tensor(targets[idx], dtype=torch.float).view(1, -1)

            if mol is None:
                continue

            xs = []
            for atom in mol.GetAtoms():
                x = []
                x.append(x_map["atomic_num"].index(atom.GetAtomicNum()))
                x.append(x_map["chirality"].index(str(atom.GetChiralTag())))
                x.append(x_map["degree"].index(atom.GetTotalDegree()))
                x.append(x_map["formal_charge"].index(atom.GetFormalCharge()))
                x.append(x_map["num_hs"].index(atom.GetTotalNumHs()))
                x.append(
                    x_map["num_radical_electrons"].index(atom.GetNumRadicalElectrons())
                )
                x.append(x_map["hybridization"].index(str(atom.GetHybridization())))
                x.append(x_map["is_aromatic"].index(atom.GetIsAromatic()))
                x.append(x_map["is_in_ring"].index(atom.IsInRing()))
                xs.append(x)

            x = torch.tensor(xs, dtype=torch.long).view(-1, 9)

            edge_indices, edge_attrs = [], []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()

                e = []
                e.append(e_map["bond_type"].index(str(bond.GetBondType())))
                e.append(e_map["stereo"].index(str(bond.GetStereo())))
                e.append(e_map["is_conjugated"].index(bond.GetIsConjugated()))

                edge_indices += [[i, j], [j, i]]
                edge_attrs += [e, e]

            edge_index = torch.tensor(edge_indices)
            edge_index = edge_index.t().to(torch.long).view(2, -1)
            edge_attr = torch.tensor(edge_attrs, dtype=torch.long).view(-1, 3)

            # Sort indices.
            if edge_index.numel() > 0:
                perm = (edge_index[0] * x.size(0) + edge_index[1]).argsort()
                edge_index, edge_attr = edge_index[:, perm], edge_attr[perm]

            data = Data(
                x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, set=set[idx]
            )

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])


def fixed_split(dataset, test_ratio):

    test_size = int(len(dataset) * test_ratio)

    # test and validation set size should be equal
    shuffled_indices = np.arange(len(dataset))
    np.random.shuffle(shuffled_indices)

    test_dataset = dataset[shuffled_indices[:test_size]]
    val_dataset = dataset[shuffled_indices[test_size : test_size * 2]]
    train_dataset = dataset[shuffled_indices[test_size * 2 :]]

    return train_dataset, val_dataset, test_dataset


class DataModule:
    """"""

    def __init__(
        self,
        data_name: str,
        root_dir: str = "data",
        data_split_mode: str = "fixed",
        test_ratio: float = 0.2,
        workers: int = 2,
        n_cv_folds: int = 10,
    ):
        """
        
        Parameters
        ----------
        data_name: str
            The name of the data that should be loaded into the datamodule. 
            Supported names are tox21, tox21_original.
        root_dir: str
            The root dir where the data is located or should be downloaded.
        data_split_mode: str
            The mode how the data should be split. Can be "fixed", "predefined", "cv" or TODO: cluster_cross_validation.
        test_ratio: float
            The percentage of the data that should be assigned to the test set and to the validation set each.
            This is only used for "fixed" data_split_mode.
        batch_size: int
            The batch size of the training data loader.
        workers: int
            The number of workers the dataloaders are using
        """

        assert data_name in _names

        self.workers = workers
        self.data_split_mode = data_split_mode

        if data_name == "tox21":
            assert data_split_mode != "predefined"

            dataset = ExtendedMoleculeNet(root="data", name="Tox21")
            self.num_classes = dataset.num_classes
            self.num_node_features = dataset.num_node_features
            self.clf_type = "multi_label"
            self.class_names = [
                "NR-AR",
                "NR-AR-LBD",
                "NR-AhR",
                "NR-Aromatase",
                "NR-ER",
                "NR-ER-LBD",
                "NR-PPAR-gamma",
                "SR-ARE",
                "SR-ATAD5",
                "SR-HSE",
                "SR-MMP",
                "SR-p53",
            ]

        if data_name == "tox21_original":

            self.clf_type = "multi_label"

            dataset = ExtendedMoleculeNet(root="data", name="tox21_original")

            info_file = pd.read_csv(
                os.path.join(root_dir, "tox21_original", "raw", "infofile.csv"),
                sep=",",
                header=0,
            )
            self.class_names = info_file.columns[-12:]

            if data_split_mode == "predefined":
                self.train_dataset = dataset[np.array(dataset.data.set) == "training"]
                self.test_dataset = dataset[np.array(dataset.data.set) == "test"]
                self.val_dataset = dataset[np.array(dataset.data.set) == "validation"]

        if data_name == "proteins":
            assert data_split_mode != "predefined"
            self.clf_type = "binary"
            self.class_names = ["Enzyme", "Non_Enzyme"]


            dataset = TUDataset(os.path.join(root_dir, "tudata"), name="PROTEINS")

        if data_name == "nci1":
            assert data_split_mode != "predefined"
            self.clf_type = "binary"
            self.class_names = ["cancer", "no_cancer"]


            dataset = TUDataset(os.path.join(root_dir, "tudata"), name="NCI1")


        self.dataset_size = len(dataset)
        self.num_classes = dataset.num_classes if self.clf_type != "binary" else 1
        self.num_node_features = dataset.num_node_features

        if data_split_mode == "fixed":
            self.train_dataset, self.val_dataset, self.test_dataset = fixed_split(
                dataset, test_ratio
            )

        if data_split_mode == "cv":
                self.dataset = dataset
                self.n_cv_folds = n_cv_folds
                self.train_indices = []
                self.test_indices = []
                self.val_indices = []
                shuffled_indices = np.arange(len(dataset))
                np.random.shuffle(shuffled_indices)

                for i in range(n_cv_folds):
                    split = len(dataset) // n_cv_folds
                    start_idx = (i * len(dataset)) // n_cv_folds
                    self.test_indices.append(shuffled_indices[start_idx: start_idx + split])
                    other = np.concatenate((shuffled_indices[:start_idx], shuffled_indices[start_idx + split:]))

                    self.val_indices.append(other[:len(dataset)//n_cv_folds])
                    self.train_indices.append(other[len(dataset)//n_cv_folds:])

    def make_train_loader(
        self,
        batch_size=64,
        use_efficient=False,
        add_self_loops=False,
        n_th_cv_fold=0,
        **kwargs,
    ):
        if self.data_split_mode == "cv":
            dataset = self.dataset[self.train_indices[n_th_cv_fold].tolist()]
        else:
            dataset = self.train_dataset

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.workers,
            collate_fn=(
                lambda batch: collate_fn_efficientoh(
                    batch, add_self_loops=add_self_loops
                )
            )
            if use_efficient
            else collate_fn,
            pin_memory=True
        )

    def make_test_loader(
        self,
        batch_size=64,
        use_efficient=False,
        add_self_loops=False,
        n_th_cv_fold=0,
        **kwargs,
    ):
        if self.data_split_mode == "cv":
            dataset = self.dataset[self.test_indices[n_th_cv_fold].tolist()]
        else:
            dataset = self.test_dataset

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.workers,
            collate_fn=(
                lambda batch: collate_fn_efficientoh(
                    batch, add_self_loops=add_self_loops
                )
            )
            if use_efficient
            else collate_fn,
            pin_memory=True
        )

    def make_val_loader(
        self,
        batch_size=64,
        use_efficient=False,
        add_self_loops=False,
        n_th_cv_fold=0,
        **kwargs,
    ):
        if self.data_split_mode == "cv":
            dataset = self.dataset[self.val_indices[n_th_cv_fold].tolist()]
        else:
            dataset = self.val_dataset

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.workers,
            collate_fn=(
                lambda batch: collate_fn_efficientoh(
                    batch, add_self_loops=add_self_loops
                )
            )
            if use_efficient
            else collate_fn,
            pin_memory=True
        )


def collate_fn(batch):

    lens = torch.tensor([len(b.x) for b in batch])
    adjs = [b.edge_index for b in batch]
    xs = [b.x for b in batch]

    col = Collater([], [])
    merged_data = col(batch)

    return merged_data, lens, adjs, xs


def collate_fn_efficientoh(batch, add_self_loops=False):

    lens = torch.tensor([len(b.x) for b in batch])

    adjs = [b.edge_index for b in batch]
    if add_self_loops:
        adjs = [
            torch_geometric.utils.add_self_loops(
                torch_geometric.utils.remove_self_loops(b)[0], num_nodes=n_nodes
            )[0]
            for b, n_nodes in zip(adjs, lens)
        ]

    adj_lens = torch.tensor([adj.shape[1] for adj in adjs])

    maxlen = lens.max()
    max_adj_len = adj_lens.max()

    dim_size = (
        maxlen + 1
    )  # we need +1 for the adjecency matrix which puts all overflowing elements to be the last node (which is a placeholder)
    adj_dim_size = max_adj_len

    adjs = torch.stack(
        [
            torch.cat(
                (
                    adj,
                    (dim_size - 1)
                    * torch.ones((2, adj_dim_size - adj.shape[1]), dtype=torch.long),
                ),
                dim=1,
            )
            for adj in adjs
        ],
        dim=0,
    )
    xs = torch.stack(
        [
            torch.vstack((b.x, torch.zeros((dim_size - b.x.shape[0], b.x.shape[1]))))
            for b in batch
        ],
        dim=0,
    )
    one_hots = torch.stack(
        [
            torch.cat(
                (
                    torch.cat((torch.eye(l), torch.zeros((dim_size - l, l))), dim=0),
                    torch.zeros(dim_size, dim_size - l),
                ),
                dim=1,
            )
            for l in lens
        ],
        dim=0,
    )

    y = torch.cat([b.y for b in batch], dim=0)

    return (
        {
            "xs": xs,
            "onehots": one_hots,
            "adjs": adjs,
            "n_nodes": lens,
            "dim_size": dim_size,
        },
        y,
    )

