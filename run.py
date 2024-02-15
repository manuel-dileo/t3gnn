from t3gnn import T3GNN
from traineval import train_roland
import argparse
import random
import torch
from torch_geometric.data import Data
import torch_geometric.transforms as T
from torch_geometric.transforms import Constant
import os
import numpy as np

# Create an ArgumentParser object
parser = argparse.ArgumentParser()


# Add command-line options
parser.add_argument('--add_self_loops', action='store_true', help='Add self loops during message-passing')
parser.add_argument('--hidden_dim', type=int, help='Size of hidden layers', default=64)
parser.add_argument('--seed', type=int, help='Random seed', default=41)

parser.add_argument('--dataset', type=str, help='Directory path of your dataset (See README)')
parser.add_argument('--num_nodes', type=int, default=None, help='Number of nodes in your dataset (used only if you do not have node features)')

args = parser.parse_args()

def load_dataset(dirname, hidden_dim, num_nodes=None):
    files = os.listdir(dirname)
    num_snap = max([int(file.split('_')[0]) for file in files])+1
    features = len([file for file in files if file.endswith("_x.pt")]) > 0
    snapshots = []
    constant = Constant()
    for i in range(num_snap):
        d = Data()
        d.edge_index = torch.load(f'{dirname}/{i}_edge_index.pt')
        if features:
            d.x = torch.load(f'{dirname}/{i}_x.pt')
        else:
            if num_nodes is None:
                raise Exception('You need to specify num_nodes if you do not have node features')
            d.num_nodes = num_nodes
            d.x = torch.randn(num_nodes, hidden_dim)
        snapshots.append(d)
    return snapshots

if __name__ == '__main__':
    seed = args.seed
    device = torch.device('cuda')
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.empty_cache()
    
    print('Loading dataset...')
    snapshots = load_dataset(args.dataset, args.hidden_dim, args.num_nodes)
    
    print('Training T3GNN...')
    scores = train_roland(snapshots, args.hidden_dim, args.hidden_dim, update='mlp', add_self_loops=args.add_self_loops)
    print('Training ended. Writing results...')
    
    if not os.path.exists(f'results-{args.dataset}'):
        os.makedirs(directory_path)
    with open(f'results-{args.dataset}/auprc_scores.txt', 'w') as wfile:
        for score in scores:
            wfile.write("%s\n" % score)
            
    print('Results saved')
    print('Done')

    
    
    
    
    
    
    
    
    