
"""
Utility classes for datasets which we test/train on
"""

__author__ = "Alexander Krauck"
__email__ = "alexander.krauck@gmail.com"
__date__ = "21-08-2021"


from  torch_geometric.data import Dataset
from torch_geometric.data import DataLoader
from torch_geometric.data import InMemoryDataset
from torch_geometric.io import parse_txt_array
from torch_geometric.data import Data

import torch
import torch.nn.functional as F
from torch_sparse import coalesce

import pandas as pd
import numpy as np

from os.path import join

from rdkit import Chem

_names = [
    "tox21",  #The tox21 dataset which is provided in the torch_geometric.datasets.MoleculeNet
    "tox21_original" #The tox21 dataset how it was originally and avaiable at
]



elems = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4, "Cl":5, "Br":6, "Na":7, "S":8, "Hg" :9, "I":10, "P":11, "Ca":12, "Pt":13, "As":14, 
"B":15, "Sn":16, "Bi":17, "Co":18, "K":19, "Fe":20, "Gd":21, "Zn":22, "Al":23, 
"Au":24, "Si":25, "Cu":26, "Cr":27, "Cd":28, "V":29, "Li":30, "Se":31, "Ag":32, "Sb":33, "Ba":34, "D":35, "Ti":36, "Tl":37, "Sr":38, "In":39, "Dy":40, "Ni":41, "Be":42,
"Mg":43, "Nd":44, "Pd":45, "Mn":46, "Zr":47, "Pb":48, "Yb":49, "Mo":50, "Ge":51, "Ru":52, "Eu":53, "Sc":54
}

import os
import os.path as osp
import re

import torch
from torch_geometric.data import (InMemoryDataset, Data, download_url,
                                  extract_gz)

try:
    from rdkit import Chem
except ImportError:
    Chem = None

x_map = {
    'atomic_num':
    list(range(0, 119)),
    'chirality': [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER',
    ],
    'degree':
    list(range(0, 11)),
    'formal_charge':
    list(range(-5, 7)),
    'num_hs':
    list(range(0, 9)),
    'num_radical_electrons':
    list(range(0, 5)),
    'hybridization': [
        'UNSPECIFIED',
        'S',
        'SP',
        'SP2',
        'SP3',
        'SP3D',
        'SP3D2',
        'OTHER',
    ],
    'is_aromatic': [False, True],
    'is_in_ring': [False, True],
}

e_map = {
    'bond_type': [
        'misc',
        'SINGLE',
        'DOUBLE',
        'TRIPLE',
        'AROMATIC',
    ],
    'stereo': [
        'STEREONONE',
        'STEREOZ',
        'STEREOE',
        'STEREOCIS',
        'STEREOTRANS',
        'STEREOANY',
    ],
    'is_conjugated': [False, True],
}


class MoleculeNet(InMemoryDataset):
    r"""The `MoleculeNet <http://moleculenet.ai/datasets-1>`_ benchmark
    collection  from the `"MoleculeNet: A Benchmark for Molecular Machine
    Learning" <https://arxiv.org/abs/1703.00564>`_ paper, containing datasets
    from physical chemistry, biophysics and physiology.
    All datasets come with the additional node and edge features introduced by
    the `Open Graph Benchmark <https://ogb.stanford.edu/docs/graphprop/>`_.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (:obj:`"ESOL"`,
            :obj:`"FreeSolv"`, :obj:`"Lipo"`, :obj:`"PCBA"`, :obj:`"MUV"`,
            :obj:`"HIV"`, :obj:`"BACE"`, :obj:`"BBPB"`, :obj:`"Tox21"`,
            :obj:`"ToxCast"`, :obj:`"SIDER"`, :obj:`"ClinTox"`).
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    """

    url = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/{}'

    # Format: name: [display_name, url_name, csv_name, smiles_idx, y_idx]
    names = {
        'esol': ['ESOL', 'delaney-processed.csv', 'delaney-processed', -1, -2],
        'freesolv': ['FreeSolv', 'SAMPL.csv', 'SAMPL', 1, 2],
        'lipo': ['Lipophilicity', 'Lipophilicity.csv', 'Lipophilicity', 2, 1],
        'pcba': ['PCBA', 'pcba.csv.gz', 'pcba', -1,
                 slice(0, 128)],
        'muv': ['MUV', 'muv.csv.gz', 'muv', -1,
                slice(0, 17)],
        'hiv': ['HIV', 'HIV.csv', 'HIV', 0, -1],
        'bace': ['BACE', 'bace.csv', 'bace', 0, 2],
        'bbbp': ['BBPB', 'BBBP.csv', 'BBBP', -1, -2],
        'tox21': ['Tox21', 'tox21.csv.gz', 'tox21', -1,
                  slice(0, 12)],
        'toxcast':
        ['ToxCast', 'toxcast_data.csv.gz', 'toxcast_data', 0,
         slice(1, 618)],
        'sider': ['SIDER', 'sider.csv.gz', 'sider', 0,
                  slice(1, 28)],
        'clintox': ['ClinTox', 'clintox.csv.gz', 'clintox', 0,
                    slice(1, 3)],
        'tox21_original': ['Tox21_original', 'tox21_original.csv.gz', 'tox21_original', -1,
                  slice(0, 12)]
    }

    def __init__(self, root, name, transform=None, pre_transform=None,
                 pre_filter=None):

        if Chem is None:
            raise ImportError('`MoleculeNet` requires `rdkit`.')

        self.name = name.lower()
        assert self.name in self.names.keys()
        super(MoleculeNet, self).__init__(root, transform, pre_transform,
                                          pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return f'{self.names[self.name][2]}.csv'

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        url = self.url.format(self.names[self.name][1])
        path = download_url(url, self.raw_dir)
        if self.names[self.name][1][-2:] == 'gz':
            extract_gz(path, self.raw_dir)
            os.unlink(path)

    def process(self):
        with open(self.raw_paths[0], 'r') as f:
            dataset = f.read().split('\n')[1:-1]
            dataset = [x for x in dataset if len(x) > 0]  # Filter empty lines.

        data_list = []
        for line in dataset:
            line = re.sub(r'\".*\"', '', line)  # Replace ".*" strings.
            line = line.split(',')

            smiles = line[self.names[self.name][3]]
            ys = line[self.names[self.name][4]]
            ys = ys if isinstance(ys, list) else [ys]

            ys = [float(y) if len(y) > 0 else float('NaN') for y in ys]
            y = torch.tensor(ys, dtype=torch.float).view(1, -1)

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue

            xs = []
            for atom in mol.GetAtoms():
                x = []
                x.append(x_map['atomic_num'].index(atom.GetAtomicNum()))
                x.append(x_map['chirality'].index(str(atom.GetChiralTag())))
                x.append(x_map['degree'].index(atom.GetTotalDegree()))
                x.append(x_map['formal_charge'].index(atom.GetFormalCharge()))
                x.append(x_map['num_hs'].index(atom.GetTotalNumHs()))
                x.append(x_map['num_radical_electrons'].index(
                    atom.GetNumRadicalElectrons()))
                x.append(x_map['hybridization'].index(
                    str(atom.GetHybridization())))
                x.append(x_map['is_aromatic'].index(atom.GetIsAromatic()))
                x.append(x_map['is_in_ring'].index(atom.IsInRing()))
                xs.append(x)

            x = torch.tensor(xs, dtype=torch.long).view(-1, 9)

            edge_indices, edge_attrs = [], []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()

                e = []
                e.append(e_map['bond_type'].index(str(bond.GetBondType())))
                e.append(e_map['stereo'].index(str(bond.GetStereo())))
                e.append(e_map['is_conjugated'].index(bond.GetIsConjugated()))

                edge_indices += [[i, j], [j, i]]
                edge_attrs += [e, e]

            edge_index = torch.tensor(edge_indices)
            edge_index = edge_index.t().to(torch.long).view(2, -1)
            edge_attr = torch.tensor(edge_attrs, dtype=torch.long).view(-1, 3)

            # Sort indices.
            if edge_index.numel() > 0:
                perm = (edge_index[0] * x.size(0) + edge_index[1]).argsort()
                edge_index, edge_attr = edge_index[:, perm], edge_attr[perm]

            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y,
                        smiles=smiles)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])

    def __repr__(self):
        return '{}({})'.format(self.names[self.name][0], len(self))


def parse_sdf(src, targets):
    graph_list = []
    mols = src.split("$$$$\n")[:-1]

    assert len(targets) == len(mols)

    for mol, y in zip(mols, targets):

        mol = mol.split('\n')[3:]

        num_atoms, num_bonds = int(mol[0][:3]), int(mol[0][3:6])#there is always a 3 digit space for them

        atom_block = mol[1:num_atoms + 1]

        pos = parse_txt_array(atom_block, end=3)

        x = torch.tensor([elems[item.split()[3]] for item in atom_block])#this could be made dynamically...
        x = F.one_hot(x, num_classes=len(elems))

        bond_block = mol[1 + num_atoms:1 + num_atoms + num_bonds]


        if len(bond_block) != 0:
            left = []
            right = []
            attr = []
            for bond_line in bond_block:
                left.append(int(bond_line[:3]) - 1)
                right.append(int(bond_line[3:6]) - 1)
                attr.append(int(bond_line[6:9]) - 1)

            left, right, attr = torch.tensor(left, dtype=torch.long), torch.tensor(right, dtype=torch.long), torch.tensor(attr, dtype=torch.long)

            left, right = torch.cat([left, right], dim=0), torch.cat([right, left], dim=0)
            edge_index = torch.stack([left, right], dim=0)
            attr = torch.cat([attr, attr], dim=0)

            edge_index, attr = coalesce(edge_index, attr, num_atoms, num_atoms)
        else:
            edge_index, attr = None, None

        graph = Data(x=x, edge_index=edge_index, edge_attr=attr, pos = pos, y=torch.tensor(y.astype(np.float32)))
        graph_list.append(graph)

    return graph_list



def read_sdf(path, targets):
    with open(path, 'r') as f:
        return parse_sdf(f.read(), targets)

class ListDataSet(Dataset):
    def __init__(self, data_list):
        self.data_list = np.array(data_list)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self.data_list[idx]
        if isinstance(idx, (slice, np.ndarray, list)):
            return ListDataSet(self.data_list[idx])

def fixed_split(dataset, test_ratio):

    test_size = int(len(dataset) * test_ratio)

    #test and validation set size should be equal
    shuffled_indices = np.arange(len(dataset))
    np.random.shuffle(shuffled_indices)

    test_dataset = dataset[shuffled_indices[:test_size]]
    val_dataset = dataset[shuffled_indices[test_size:test_size * 2]]
    train_dataset = dataset[shuffled_indices[test_size * 2:]]

    return train_dataset, val_dataset, test_dataset

class DataModule():
    """"""

    def __init__(self, data_name: str, root_dir: str = "data", split_mode: str = "fixed", test_ratio: float = 0.2, batch_size: int = 64):
        """
        
        Parameters
        ----------
        data_name: str
            The name of the data that should be loaded into the datamodule. 
            Supported names are tox21, tox21_original.
        root_dir: str
            The root dir where the data is located or should be downloaded.
        split_mode: str
            The mode how the data should be split. Can be "fixed", "predefined" or TODO: cluster_cross_validation.
        test_ratio: float
            The percentage of the data that should be assigned to the test set and to the validation set each.
            This is only used for "fixed" split_mode.
        batch_size: int
            The batch size of the training data loader.
        """
        
        assert data_name in _names

        if data_name == "tox21":
            dataset = MoleculeNet(root="data", name="Tox21")
            self.dataset_size = len(dataset)
            self.num_classes = dataset.num_classes
            self.num_node_features = dataset.num_node_features
            self.class_names = ["NR-AR","NR-AR-LBD","NR-AhR","NR-Aromatase","NR-ER","NR-ER-LBD","NR-PPAR-gamma","SR-ARE","SR-ATAD5","SR-HSE","SR-MMP","SR-p53"]


            if split_mode == "fixed":
                train_dataset, val_dataset, test_dataset = fixed_split(dataset, test_ratio)
    
            self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            self.test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
            self.val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

        if data_name == "tox21_original":

            dataset = MoleculeNet(root = "data", name = "tox21_original")

            self.dataset_size = len(dataset)
            self.num_classes = dataset.num_classes
            self.num_node_features = dataset.num_node_features
            info_file = pd.read_csv(join(root_dir,"tox21_original","infofile.csv"), sep=",", header=0)

            self.class_names = info_file.columns[-12:]

            if split_mode == "fixed":
                train_dataset, val_dataset, test_dataset = fixed_split(dataset, test_ratio)

            elif split_mode == "predefined":
                set_type = info_file.reset_index()

                training_rows = set_type.index[set_type["set"] == "training"].to_numpy()
                validation_rows = set_type.index[set_type["set"] == "validation"].to_numpy()
                test_rows = set_type.index[set_type["set"] == "test"].to_numpy()

                train_dataset = dataset[training_rows - 1]
                val_dataset = dataset[validation_rows - 1]
                test_dataset = dataset[test_rows - 1]

            self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            self.test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
            self.val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)







