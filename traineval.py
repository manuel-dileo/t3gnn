#THIS FILE CONTAINS TRAINING AND TEST UTILITY METHODS
import torch
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, GRUCell
from torch_geometric.data import Data
from sklearn.metrics import roc_auc_score,average_precision_score

import random

import bisect 

import gc
import copy

from itertools import permutations

import pandas as pd

from torch_geometric.utils import negative_sampling, erdos_renyi_graph, shuffle_node, to_networkx
import torch_geometric.transforms as T
from torch_geometric.utils import train_test_split_edges
from torch_geometric.transforms import RandomLinkSplit,NormalizeFeatures,Constant,OneHotDegree
from torch_geometric.utils import from_networkx
from torch_geometric.nn import GCNConv,SAGEConv,GATv2Conv, GINConv, Linear
from scipy.stats import entropy

import torch
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

import copy
import itertools
import json
from steemitdata import get_steemit_dataset

from t3gnn import T3GConvGRU, T3EvolveGCNH, T3EvolveGCNO, T3GNN, T3MLP

def roland_test(model, test_data, data, isnap, device='cpu'):
    model.eval()

    test_data = test_data.to(device)

    h, _ = model(test_data.x, test_data.edge_index, edge_label_index = test_data.edge_label_index, isnap=isnap)
    
    pred_cont_link = torch.sigmoid(h).cpu().detach().numpy()
    
    label_link = test_data.edge_label.cpu().detach().numpy()
      
    avgpr_score_link = average_precision_score(label_link, pred_cont_link)
    
    return avgpr_score_link

def ev_test(model, test_data, data, device='cpu'): 
    model.eval()
    test_data = test_data.to(device)
    h = model(test_data.x, test_data.edge_index, test_data.edge_label_index)
    pred_cont = torch.sigmoid(h).cpu().detach().numpy()
    label = test_data.edge_label.cpu().detach().numpy()
    avgpr_score = average_precision_score(label, pred_cont)
    return avgpr_score

def gcgru_test(model, test_data, data, device='cpu'):
    model.eval()
    test_data = test_data.to(device)
    h, _ = model(test_data.x, test_data.edge_index, test_data.edge_label_index)
    pred_cont = torch.sigmoid(h).cpu().detach().numpy()
    label = test_data.edge_label.cpu().detach().numpy()
    avgpr_score = average_precision_score(label, pred_cont)
    return avgpr_score

from sklearn.metrics import *

def roland_train_single_snapshot(model, data, train_data, val_data, test_data, isnap,\
                          last_embeddings, optimizer, device='cpu', num_epochs=50, verbose=False):
    
    avgpr_val_max = 0
    best_model = model
    train_data = train_data.to(device)
    best_epoch = -1
    best_current_embeddings = []
    
    avgpr_trains = []
    #avgpr_vals = []
    avgpr_tests = []
    
    tol = 1
    
    for epoch in range(num_epochs):
        model.train()
        ## Note
        ## 1. Zero grad the optimizer
        ## 2. Compute loss and backpropagate
        ## 3. Update the model parameters
        optimizer.zero_grad()

        pred,\
        current_embeddings =\
            model(train_data.x, train_data.edge_index, edge_label_index = train_data.edge_label_index,\
                  isnap=isnap, previous_embeddings=last_embeddings)
        
        loss = model.loss(pred, train_data.edge_label.type_as(pred)) #loss to fine tune on current snapshot

        loss.backward(retain_graph=True)  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.

        ##########################################

        log = 'Epoch: {:03d}\n AVGPR Train: {:.4f}, Val: {:.4f}, Test: {:.4f}\n MRR Train: {:.4f}, Val: {:.4f}, Test: {:.4f}\n F1-Score Train: {:.4f}, Val: {:.4f}, Test: {:.4f}\n Loss: {}'
        avgpr_score_val  = roland_test(model, val_data, data, isnap, device)
        
        if avgpr_val_max-tol <= avgpr_score_val:
            avgpr_val_max = avgpr_score_val
            best_epoch = epoch
            best_current_embeddings = current_embeddings
            best_model = model
        else:
            break
        
        
    #avgpr_score_train = roland_test(model, train_data, data, isnap, device)
    avgpr_score_test = roland_test(model, test_data, data, isnap, device)
            
    if verbose:
        print(f'Best Epoch: {best_epoch}')
    #print(f'Best Epoch: {best_epoch}')
    
    return best_model, optimizer, avgpr_score_test, best_current_embeddings

def gcgru_train_single_snapshot(model, data, train_data, val_data, test_data,\
                          optimizer, H=None, device='cpu', num_epochs=50, verbose=False):
    
    mrr_val_max = 0
    avgpr_val_max = 0
    best_model = model
    train_data = train_data.to(device)
    best_epoch = -1
    
    tol = 5e-2
    
    best_H = None
    
    for epoch in range(num_epochs):
        model.train()
        ## Note
        ## 1. Zero grad the optimizer
        ## 2. Compute loss and backpropagate
        ## 3. Update the model parameters
        optimizer.zero_grad()
        
        #H = None
            
        pred, H = model(train_data.x, train_data.edge_index, train_data.edge_label_index, H)
        
        loss = model.loss(pred, train_data.edge_label.type_as(pred)) #loss to fine tune on current snapshot

        loss.backward(retain_graph=True)  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.

        ##########################################

        log = 'Epoch: {:03d}\n AVGPR Train: {:.4f}, Val: {:.4f}, Test: {:.4f}\n MRR Train: {:.4f}, Val: {:.4f}, Test: {:.4f}\n F1-Score Train: {:.4f}, Val: {:.4f}, Test: {:.4f}\n Loss: {}'
        avgpr_score_val = gcgru_test(model, val_data, data, device)
        
        """
        if mrr_val_max-tol < mrr_val:
            mrr_val_max = mrr_val
            best_epoch = epoch
            best_current_embeddings = current_embeddings
            best_model = copy.deepcopy(model)
        else:
            break
        
        #print(f'Epoch: {epoch} done')
            
        """
        if avgpr_val_max-tol <= avgpr_score_val:
            avgpr_val_max = avgpr_score_val
            best_H = H.clone()
            best_epoch = epoch
            best_model = model
        else:
            break
        
    avgpr_score_test = gcgru_test(model, test_data, data, device)
            
    if verbose:
        print(f'Best Epoch: {best_epoch}')
    #print(f'Best Epoch: {best_epoch}')
    
    return best_model, avgpr_score_test, best_H, optimizer

def ev_train_single_snapshot(model, data, train_data, val_data, test_data,\
                          optimizer, device='cpu', num_epochs=50, verbose=False):
    
    mrr_val_max = 0
    avgpr_val_max = 0
    best_model = model
    train_data = train_data.to(device)
    best_epoch = -1
    
    tol = 5e-2
    
    for epoch in range(num_epochs):
        model.train()
        ## Note
        ## 1. Zero grad the optimizer
        ## 2. Compute loss and backpropagate
        ## 3. Update the model parameters
        optimizer.zero_grad()
            
        pred = model(train_data.x, train_data.edge_index, train_data.edge_label_index)
        
        loss = model.loss(pred, train_data.edge_label.type_as(pred)) #loss to fine tune on current snapshot

        loss.backward(retain_graph=True)  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.

        ##########################################

        log = 'Epoch: {:03d}\n AVGPR Train: {:.4f}, Val: {:.4f}, Test: {:.4f}\n MRR Train: {:.4f}, Val: {:.4f}, Test: {:.4f}\n F1-Score Train: {:.4f}, Val: {:.4f}, Test: {:.4f}\n Loss: {}'
        avgpr_score_val = ev_test(model, val_data, data, device)
        
        """
        if mrr_val_max-tol < mrr_val:
            mrr_val_max = mrr_val
            besst_epoch = epoch
            best_current_embeddings = current_embeddings
            best_model = copy.deepcopy(model)
        else:
            break
        
        #print(f'Epoch: {epoch} done')
            
        """
        if avgpr_val_max-tol <= avgpr_score_val:
            avgpr_val_max = avgpr_score_val
            best_epoch = epoch
            best_model = model
        else:
            break
        
    avgpr_score_test = ev_test(model, test_data, data, device)
            
    if verbose:
        print(f'Best Epoch: {best_epoch}')
    #print(f'Best Epoch: {best_epoch}')
    
    return best_model, avgpr_score_test, optimizer

def train_models(snapshots, hidden_conv1, hidden_conv2, update='gru', device='cpu'):
    """
        Train and evaluate all the baselines in the live update setting
    """
    num_snap = len(snapshots)
    input_channels = snapshots[0].x.size(1)
    num_nodes = snapshots[0].x.size(0)
    last_embeddings = [torch.Tensor([[0 for i in range(hidden_conv1)] for j in range(num_nodes)]),\
                                    torch.Tensor([[0 for i in range(hidden_conv2)] for j in range(num_nodes)])]
    
    #TODO: rifare per ogni modello
    ro_avgpr_test_singles = []
    gcgru_avgpr_test_singles = []
    evo_avgpr_test_singles = []
    evh_avgpr_test_singles = []
    
    roland = T3GNN(input_channels, 2, hidden_conv1, dropout=0.3, update=update)
    rolopt = torch.optim.Adam(params=roland.parameters(), lr=0.01, weight_decay = 5e-3)
    roland.reset_parameters()
    
    gcgru = T3GConvGRU(input_channels, hidden_conv2)
    gcgruopt = torch.optim.Adam(params=gcgru.parameters(), lr=0.01, weight_decay = 5e-3)
    gcgru.reset_parameters()
    H = None
    
    evh = T3EvolveGCNH(num_nodes, input_channels)
    evhopt = torch.optim.Adam(params=evh.parameters(), lr=0.01, weight_decay = 5e-3)
    evh.reset_parameters()
    
    evo = T3EvolveGCNO(input_channels)
    evopt = torch.optim.Adam(params=evo.parameters(), lr=0.01, weight_decay = 5e-3)
    evo.reset_parameters()
    
    for i in range(num_snap-1):
        #CREATE TRAIN + VAL + TEST SET FOR THE CURRENT SNAP
        snapshot = copy.deepcopy(snapshots[i])
        num_current_edges = len(snapshot.edge_index[0])
        transform = RandomLinkSplit(num_val=0.0,num_test=0.25)
        train_data, _, val_data = transform(snapshot)
        test_data = copy.deepcopy(snapshots[i+1])
        
        #NEGATIVE SET: EDGES CLOSED IN THE PAST BUT NON IN THE CURRENT TEST SET
        past_edges = set(zip([int(e) for e in snapshot.edge_index[0]],\
                             [int(e) for e in snapshot.edge_index[1]]))
        current_edges = set(zip([int(e) for e in test_data.edge_index[0]],\
                             [int(e) for e in test_data.edge_index[1]]))
        
        negative_edges = list(past_edges.difference(current_edges))[:test_data.edge_index.size(1)]
        future_neg_edge_index = torch.Tensor([[a[0] for a in negative_edges],\
                                                 [a[1] for a in negative_edges]]).long()
        
        num_pos_edge = test_data.edge_index.size(1)
        num_neg_edge = future_neg_edge_index.size(1)
        test_data.edge_label = torch.Tensor(np.array([1 for i in range(num_pos_edge)] + [0 for i in range(num_neg_edge)]))
        test_data.edge_label_index = torch.cat([test_data.edge_index, future_neg_edge_index], dim=-1)
        
        #TRAIN AND TEST THE MODELS FOR THE CURRENT SNAP
        roland, rolopt, ro_avgpr_test, last_embeddings =\
            roland_train_single_snapshot(roland, snapshot, train_data, val_data, test_data, i,\
                                  last_embeddings, rolopt)
        
        gcgru, gcgru_avgpr_test, H, gcgruopt =\
            gcgru_train_single_snapshot(gcgru, snapshot, train_data, val_data, test_data, gcgruopt, H)
        
        evo, evo_avgpr_test, evopt =\
            ev_train_single_snapshot(evo, snapshot, train_data, val_data, test_data, evopt)
        
        evh, evh_avgpr_test, evhopt =\
            ev_train_single_snapshot(evh, snapshot, train_data, val_data, test_data, evhopt)
        
        
        #SAVE AND DISPLAY EVALUATION
        print(f'Snapshot: {i}\n\tT3GNN AVGPR Test: {ro_avgpr_test}')
        print(f'\tGCGRU AVGPR Test: {gcgru_avgpr_test}')
        print(f'\tEvolveGCN-O AVGPR Test: {evo_avgpr_test}')
        print(f'\tEvolveGCN-H AVGPR Test: {evh_avgpr_test}')
        
        ro_avgpr_test_singles.append(ro_avgpr_test)
        gcgru_avgpr_test_singles.append(gcgru_avgpr_test)
        evo_avgpr_test_singles.append(evo_avgpr_test)
        evh_avgpr_test_singles.append(evh_avgpr_test)
        
    ro_avgpr_test_all = sum(ro_avgpr_test_singles)/len(ro_avgpr_test_singles)
    gcgru_avgpr_test_all = sum(gcgru_avgpr_test_singles)/len(gcgru_avgpr_test_singles)
    evo_avgpr_test_all = sum(evo_avgpr_test_singles)/len(evo_avgpr_test_singles)
    evh_avgpr_test_all = sum(evh_avgpr_test_singles)/len(evh_avgpr_test_singles)
    
    print(f'T3GNN AVGPR over time: Test: {ro_avgpr_test_all}')
    print(f'GCGRU AVGPR over time: Test: {gcgru_avgpr_test_all}')
    print(f'EvolveGCN-O AVGPR over time: Test: {evo_avgpr_test_all}')
    print(f'EvolveGCN-H AVGPR over time: Test: {evh_avgpr_test_all}')
    
    return ro_avgpr_test_singles, gcgru_avgpr_test_singles, evo_avgpr_test_singles, evh_avgpr_test_singles

def train_roland(snapshots, hidden_conv1, hidden_conv2, update='gru', device='cpu',\
                 add_self_loops=False, skip_connections=False, content_mlp=False,\
                 shuffle_node_features=False, random_graph=False):
    """
        Train and evaluate T3GNN with historical negative edges in the live update setting
    """
    num_snap = len(snapshots)
    input_channels = snapshots[0].x.size(1)
    num_nodes = snapshots[0].x.size(0)
    last_embeddings = [torch.Tensor([[0 for i in range(hidden_conv1)] for j in range(num_nodes)]),\
                                    torch.Tensor([[0 for i in range(hidden_conv2)] for j in range(num_nodes)])]
 
    avgpr_test_singles = []
    
    hidden_dimension = hidden_conv1
    
    roland = T3GNN(input_channels, 2, hidden_dimension, dropout=0.3, update=update,\
                  add_self_loops = add_self_loops, skip_connections=skip_connections, content_mlp=content_mlp)
    rolopt = torch.optim.Adam(params=roland.parameters(), lr=0.01, weight_decay = 5e-3)
    roland.reset_parameters()
    
    for i in range(num_snap-1):
        #CREATE TRAIN + VAL + TEST SET FOR THE CURRENT SNAP
        snapshot = copy.deepcopy(snapshots[i])
        if shuffle_node_features:
            snapshot.x, _ = shuffle_node(snapshot.x)
        num_current_edges = len(snapshot.edge_index[0])
        transform = RandomLinkSplit(num_val=0.0,num_test=0.25)
        train_data, _, val_data = transform(snapshot)
        if random_graph:
            density = snapshot.edge_index.size(0) / (num_nodes * (num_nodes-1))
            train_data.edge_index = erdos_renyi_graph(num_nodes, density, directed=True)
        test_data = copy.deepcopy(snapshots[i+1])
        
        #NEGATIVE SET: EDGES CLOSED IN THE PAST BUT NON IN THE CURRENT TEST SET
        past_edges = set(zip([int(e) for e in snapshot.edge_index[0]],\
                             [int(e) for e in snapshot.edge_index[1]]))
        current_edges = set(zip([int(e) for e in test_data.edge_index[0]],\
                             [int(e) for e in test_data.edge_index[1]]))
        
        negative_edges = list(past_edges.difference(current_edges))[:test_data.edge_index.size(1)]
        future_neg_edge_index = torch.Tensor([[a[0] for a in negative_edges],\
                                                 [a[1] for a in negative_edges]]).long()
        
        num_pos_edge = test_data.edge_index.size(1)
        num_neg_edge = future_neg_edge_index.size(1)
        test_data.edge_label = torch.Tensor(np.array([1 for i in range(num_pos_edge)] + [0 for i in range(num_neg_edge)]))
        test_data.edge_label_index = torch.cat([test_data.edge_index, future_neg_edge_index], dim=-1)
        
        #TRAIN AND TEST THE MODEL FOR THE CURRENT SNAP
        roland, rolopt, avgpr_test, last_embeddings =\
            roland_train_single_snapshot(roland, snapshot, train_data, val_data, test_data, i,\
                                  last_embeddings, rolopt)
        
        
        #SAVE AND DISPLAY EVALUATION
        print(f'Snapshot: {i}\n\tT3GNN AVGPR Test: {avgpr_test}')
        avgpr_test_singles.append(avgpr_test)
        
    avgpr_test_all = sum(avgpr_test_singles)/len(avgpr_test_singles)
    
    print(f'T3GNN AVGPR over time Test: {avgpr_test_all}')
    
    return avgpr_test_singles

def train_mlp(snapshots, hidden_conv1, hidden_conv2, update='gru', device='cpu'):
    """
        Train and evaluate T3GNN with historical negative edges in the live update setting
    """
    num_snap = len(snapshots)
    input_channels = snapshots[0].x.size(1)
    num_nodes = snapshots[0].x.size(0)
    last_embeddings = [torch.Tensor([[0 for i in range(hidden_conv1)] for j in range(num_nodes)]),\
                                    torch.Tensor([[0 for i in range(hidden_conv2)] for j in range(num_nodes)])]
 
    avgpr_test_singles = []
    
    hidden_dimension = hidden_conv1
    
    roland = T3MLP(input_channels, 2, hidden_dimension, dropout=0.3, update=update)
    rolopt = torch.optim.Adam(params=roland.parameters(), lr=0.01, weight_decay = 5e-3)
    roland.reset_parameters()
    
    for i in range(num_snap-1):
        #CREATE TRAIN + VAL + TEST SET FOR THE CURRENT SNAP
        snapshot = copy.deepcopy(snapshots[i])
        num_current_edges = len(snapshot.edge_index[0])
        transform = RandomLinkSplit(num_val=0.0,num_test=0.25)
        train_data, _, val_data = transform(snapshot)
        test_data = copy.deepcopy(snapshots[i+1])
        
        #NEGATIVE SET: EDGES CLOSED IN THE PAST BUT NON IN THE CURRENT TEST SET
        past_edges = set(zip([int(e) for e in snapshot.edge_index[0]],\
                             [int(e) for e in snapshot.edge_index[1]]))
        current_edges = set(zip([int(e) for e in test_data.edge_index[0]],\
                             [int(e) for e in test_data.edge_index[1]]))
        
        negative_edges = list(past_edges.difference(current_edges))[:test_data.edge_index.size(1)]
        future_neg_edge_index = torch.Tensor([[a[0] for a in negative_edges],\
                                                 [a[1] for a in negative_edges]]).long()
        
        num_pos_edge = test_data.edge_index.size(1)
        num_neg_edge = future_neg_edge_index.size(1)
        test_data.edge_label = torch.Tensor(np.array([1 for i in range(num_pos_edge)] + [0 for i in range(num_neg_edge)]))
        test_data.edge_label_index = torch.cat([test_data.edge_index, future_neg_edge_index], dim=-1)
        
        #TRAIN AND TEST THE MODEL FOR THE CURRENT SNAP
        roland, rolopt, avgpr_test, last_embeddings =\
            roland_train_single_snapshot(roland, snapshot, train_data, val_data, test_data, i,\
                                  last_embeddings, rolopt)
        
        
        #SAVE AND DISPLAY EVALUATION
        print(f'Snapshot: {i}\n\tT3GNN AVGPR Test: {avgpr_test}')
        avgpr_test_singles.append(avgpr_test)
        
    avgpr_test_all = sum(avgpr_test_singles)/len(avgpr_test_singles)
    
    print(f'T3MLP AVGPR over time Test: {avgpr_test_all}')
    
    return avgpr_test_singles

def edge_bank(snapshots):
    """
       Edgebank baseline
    """
    num_snap = len(snapshots)
    num_nodes = snapshots[0].x.size(0)
 
    avgpr_test_singles = []
    
    edge_bank = nx.DiGraph()
    
    for i in range(num_snap-1):
        #CREATE TRAIN + VAL + TEST SET FOR THE CURRENT SNAP
        snapshot = copy.deepcopy(snapshots[i])
        num_current_edges = len(snapshot.edge_index[0])
        transform = RandomLinkSplit(num_val=0.0,num_test=0.25)
        train_data, _, val_data = transform(snapshot)
        test_data = copy.deepcopy(snapshots[i+1])
        
        #NEGATIVE SET: EDGES CLOSED IN THE PAST BUT NON IN THE CURRENT TEST SET
        past_edges = set(zip([int(e) for e in snapshot.edge_index[0]],\
                             [int(e) for e in snapshot.edge_index[1]]))
        current_edges = set(zip([int(e) for e in test_data.edge_index[0]],\
                             [int(e) for e in test_data.edge_index[1]]))
        
        negative_edges = list(past_edges.difference(current_edges))[:test_data.edge_index.size(1)]
        future_neg_edge_index = torch.Tensor([[a[0] for a in negative_edges],\
                                                 [a[1] for a in negative_edges]]).long()
        
        num_pos_edge = test_data.edge_index.size(1)
        num_neg_edge = future_neg_edge_index.size(1)
        test_data.edge_label = torch.Tensor(np.array([1 for i in range(num_pos_edge)] + [0 for i in range(num_neg_edge)]))
        test_data.edge_label_index = torch.cat([test_data.edge_index, future_neg_edge_index], dim=-1)
        
        #TRAIN AND TEST THE MODEL FOR THE CURRENT SNAP
        #to networkx train_data, then add to existing edges, then pred with an if
        to_add_edges = list(to_networkx(train_data).edges())
        edge_bank.add_edges_from(to_add_edges)
        
        pred_cont = []
        for src,dst in zip(test_data.edge_label_index[0].detach().numpy(),test_data.edge_label_index[1].detach().numpy()):
            pred_cont.append(1 if edge_bank.has_edge(src,dst) else 0)
        label = test_data.edge_label.cpu().detach().numpy()
        avgpr_test = average_precision_score(label, pred_cont)
        
        #SAVE AND DISPLAY EVALUATION
        print(f'Snapshot: {i}\n\tEdgeBank AVGPR Test: {avgpr_test}')
        avgpr_test_singles.append(avgpr_test)
        
    avgpr_test_all = sum(avgpr_test_singles)/len(avgpr_test_singles)
    
    print(f'EdgeBank AVGPR over time Test: {avgpr_test_all}')
    
    return avgpr_test_singles

def train_roland_random(snapshots, hidden_conv1, hidden_conv2, update='gru', device='cpu'):
    """
        Train T3GNN with random negative sampling
    """
    num_snap = len(snapshots)
    input_channels = snapshots[0].x.size(1)
    num_nodes = snapshots[0].x.size(0)
    last_embeddings = [torch.Tensor([[0 for i in range(hidden_conv1)] for j in range(num_nodes)]),\
                                    torch.Tensor([[0 for i in range(hidden_conv2)] for j in range(num_nodes)])]
 
    avgpr_test_singles = []
    
    roland = T3GNN(input_channels, 2, hidden_conv1, dropout=0.3, update=update)
    rolopt = torch.optim.Adam(params=roland.parameters(), lr=0.01, weight_decay = 5e-3)
    roland.reset_parameters()
    
    for i in range(num_snap-1):
        #CREATE TRAIN + VAL + TEST SET FOR THE CURRENT SNAP
        snapshot = copy.deepcopy(snapshots[i])
        num_current_edges = len(snapshot.edge_index[0])
        transform = RandomLinkSplit(num_val=0.0,num_test=0.25)
        train_data, _, val_data = transform(snapshot)
        test_data = copy.deepcopy(snapshots[i+1])
        
        future_neg_edge_index = negative_sampling(
            edge_index=test_data.edge_index, #positive edges
            num_nodes=test_data.num_nodes, # number of nodes
            num_neg_samples=test_data.edge_index.size(1)) # number of neg_sample equal to number of pos_edges
        #edge index ok, edge_label concat, edge_label_index concat
        num_pos_edge = test_data.edge_index.size(1)
        test_data.edge_label = torch.Tensor(np.array([1 for i in range(num_pos_edge)] + [0 for i in range(num_pos_edge)]))
        test_data.edge_label_index = torch.cat([test_data.edge_index, future_neg_edge_index], dim=-1)
        
        #TRAIN AND TEST THE MODEL FOR THE CURRENT SNAP
        roland, rolopt, avgpr_test, last_embeddings =\
            roland_train_single_snapshot(roland, snapshot, train_data, val_data, test_data, i,\
                                  last_embeddings, rolopt)
        
        
        #SAVE AND DISPLAY EVALUATION
        print(f'Snapshot: {i}\n\tT3GNN random sampling AVGPR Test: {avgpr_test}')
        avgpr_test_singles.append(avgpr_test)
        
    avgpr_test_all = sum(avgpr_test_singles)/len(avgpr_test_singles)
    
    print(f'T3GNN AVGPR over time Test: {avgpr_test_all}')
    
    return avgpr_test_singles