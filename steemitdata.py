import torch
from torch_geometric.data import Data

import torch_geometric.transforms as T

from torch_geometric.transforms import Constant

def get_steemit_dataset(preprocess='constant'):
    num_snap = 26
    num_nodes = 14814
    snapshots = []
    constant = Constant()
    for i in range(num_snap):
        d = Data()
        d.num_nodes = num_nodes
        d.edge_index = torch.load(f'steemit-t3gnn-data/{i}_edge_index.pt')
        if preprocess=='constant':
            d = constant(d)
        else:
            d.x = torch.load(f'steemit-t3gnn-data/{i}_x.pt')
        snapshots.append(d)
    return snapshots