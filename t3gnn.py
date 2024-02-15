import torch
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, GRUCell, CrossEntropyLoss
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops
from sklearn.metrics import roc_auc_score,average_precision_score

import random

import copy

import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv,SAGEConv,GATv2Conv, GINConv, Linear

import torch
import networkx as nx
import numpy as np

from torch_geometric_temporal.nn.recurrent import GConvGRU,  EvolveGCNH, EvolveGCNO
from lin_rnn import LinRNN
import torch.nn as nn


class T3GNN(torch.nn.Module):
    def __init__(self, input_dim, num_gnn_layers, hidden_dim, dropout=0.0,\
                 update='mlp', loss=BCEWithLogitsLoss,\
                 add_self_loops=False, skip_connections=False, content_mlp=False):
        
        super(T3GNN, self).__init__()
        self.preprocess1 = Linear(input_dim, 256)
        self.preprocess2 = Linear(256, 128)
        self.convs = nn.ModuleList()
        if skip_connections:
            hidden_dim = 128
        self.convs.append(GCNConv(128, hidden_dim))
        for _ in range(num_gnn_layers-1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.postprocess1 = Linear(hidden_dim, 2)
        
        #Initialize the loss function to BCEWithLogitsLoss
        self.loss_fn = loss()

        self.dropout = dropout
        self.update = update
        self.num_gnn_layers = num_gnn_layers
        
        self.add_self_loops=add_self_loops
        self.skip_connections = skip_connections
        self.content_mlp = content_mlp
        
        if self.content_mlp:
            self.content_nn = nn.ModuleList()
            self.content_nn.append(Linear(input_dim, hidden_dim))
            for _ in range(num_gnn_layers-1):
                self.content_nn.append(Linear(hidden_dim, hidden_dim))
        
        self.updates = nn.ModuleList()
        if update=='avg':
            self.tau0 = torch.nn.Parameter(torch.Tensor([0.2]))
        else:
            for _ in range(num_gnn_layers):
                if update=='gru':
                    self.updates.append(GRUCell(hidden_dim, hidden_dim))
                elif update=='mlp':
                    self.updates.append(Linear(hidden_dim*2, hidden_dim))
                elif update=='linrnn':
                    self.updates.append(LinRNN(hidden_dim, hidden_dim))
        self.previous_embeddings = None
                                    
        
    def reset_loss(self,loss=BCEWithLogitsLoss):
        self.loss_fn = loss()
        
    def reset_parameters(self):
        self.preprocess1.reset_parameters()
        self.preprocess2.reset_parameters()
        for i in range(self.num_gnn_layers):
            self.convs[i].reset_parameters()
            self.updates[i].reset_parameters()
            if self.content_mlp:
                self.content_nn[i].reset_parameters()
        self.postprocess1.reset_parameters()
            
        

    def forward(self, x, edge_index, edge_label_index=None, isnap=0, previous_embeddings=None):
        
        #You do not need all the parameters to be different to None in test phase
        #You can just use the saved previous embeddings and tau
        if previous_embeddings is not None and isnap > 0: #None if test
            self.previous_embeddings = [previous_embeddings[i].clone() for i in range(self.num_gnn_layers)]
            
        current_embeddings = [torch.Tensor([]) for i in range(self.num_gnn_layers)]
        
        if self.add_self_loops:
            edge_index, _ = add_self_loops(edge_index)
            edge_index = edge_index.long()
        
        #Preprocess node repr
        h = self.preprocess1(x)
        h = h.relu()
        h = torch.Tensor(F.dropout(h, p=self.dropout).detach().numpy())
        h = self.preprocess2(h)
        h = h.relu()
        h = torch.Tensor(F.dropout(h, p=self.dropout).detach().numpy())
        h_skip = h.clone()
        
        #ROLAND forward
        for z in range(self.num_gnn_layers):
            h = self.convs[z](h, edge_index)
            h = h.relu()
            h = torch.Tensor(F.dropout(h, p=self.dropout).detach().numpy())
            #Embedding Update after first layer
            if isnap > 0:
                if self.update=='gru' or self.update=='linrnn':
                    h = torch.Tensor(self.updates[z](h, self.previous_embeddings[z].clone()).detach().numpy())
                elif self.update=='mlp':
                    hin = torch.cat((h,self.previous_embeddings[z].clone()),dim=1)
                    h = torch.Tensor(self.updates[z](hin).detach().numpy())
                else:
                    h = torch.Tensor((self.tau0 * self.previous_embeddings[z].clone() + (1-self.tau0) * h.clone()).detach().numpy())
            if self.skip_connections:
                h = torch.Tensor(torch.mul(h_skip.clone(),h.clone()))
            current_embeddings[z] = h.clone()
            
        if self.content_mlp:
            h_content = x.clone()
            for z in range(self.num_gnn_layers):
                h_content = self.content_nn[z](h_content)
                h_content = h_content.relu()
            h = torch.Tensor(torch.mul(h.clone(), h_content.clone()))
        
        #HADAMARD MLP
        h_src = h[edge_label_index[0]]
        h_dst = h[edge_label_index[1]]
        h_hadamard = torch.mul(h_src, h_dst) #hadamard product
        h = self.postprocess1(h_hadamard)
        h = torch.sum(h.clone(), dim=-1).clone()
        
        #return both 
        #i)the predictions for the current snapshot 
        #ii) the embeddings of current snapshot

        return h, current_embeddings
    
    def loss(self, pred, link_label):
        return self.loss_fn(pred, link_label)
    
class T3MLP(torch.nn.Module):
    def __init__(self, input_dim, num_gnn_layers, hidden_dim, dropout=0.0,\
                 update='mlp', loss=BCEWithLogitsLoss):
        
        super(T3MLP, self).__init__()
        self.preprocess1 = Linear(input_dim, 256)
        self.preprocess2 = Linear(256, 128)
        self.convs = nn.ModuleList()
        self.convs.append(Linear(128, hidden_dim))
        for _ in range(num_gnn_layers-1):
            self.convs.append(Linear(hidden_dim, hidden_dim))
        self.postprocess1 = Linear(hidden_dim, 2)
        
        #Initialize the loss function to BCEWithLogitsLoss
        self.loss_fn = loss()

        self.dropout = dropout
        self.update = update
        self.num_gnn_layers = num_gnn_layers
        
        self.updates = nn.ModuleList()
        if update=='avg':
            self.tau0 = torch.nn.Parameter(torch.Tensor([0.2]))
        else:
            for _ in range(num_gnn_layers):
                if update=='gru':
                    self.updates.append(GRUCell(hidden_dim, hidden_dim))
                elif update=='mlp':
                    self.updates.append(Linear(hidden_dim*2, hidden_dim))
                elif update=='linrnn':
                    self.updates.append(LinRNN(hidden_dim, hidden_dim))
        self.previous_embeddings = None
                                    
        
    def reset_loss(self,loss=BCEWithLogitsLoss):
        self.loss_fn = loss()
        
    def reset_parameters(self):
        self.preprocess1.reset_parameters()
        self.preprocess2.reset_parameters()
        for i in range(self.num_gnn_layers):
            self.convs[i].reset_parameters()
            self.updates[i].reset_parameters()
        self.postprocess1.reset_parameters()
            
        

    def forward(self, x, edge_index, edge_label_index=None, isnap=0, previous_embeddings=None):
        
        #You do not need all the parameters to be different to None in test phase
        #You can just use the saved previous embeddings and tau
        if previous_embeddings is not None and isnap > 0: #None if test
            self.previous_embeddings = [previous_embeddings[i].clone() for i in range(self.num_gnn_layers)]
            
        current_embeddings = [torch.Tensor([]) for i in range(self.num_gnn_layers)]
        
        #Preprocess node repr
        h = self.preprocess1(x)
        h = h.relu()
        h = torch.Tensor(F.dropout(h, p=self.dropout).detach().numpy())
        h = self.preprocess2(h)
        h = h.relu()
        h = torch.Tensor(F.dropout(h, p=self.dropout).detach().numpy())
        h_skip = h.clone()
        
        #ROLAND forward
        for z in range(self.num_gnn_layers):
            h = self.convs[z](h)
            h = h.relu()
            h = torch.Tensor(F.dropout(h, p=self.dropout).detach().numpy())
            #Embedding Update after first layer
            if isnap > 0:
                if self.update=='gru' or self.update=='linrnn':
                    h = torch.Tensor(self.updates[z](h, self.previous_embeddings[z].clone()).detach().numpy())
                elif self.update=='mlp':
                    hin = torch.cat((h,self.previous_embeddings[z].clone()),dim=1)
                    h = torch.Tensor(self.updates[z](hin).detach().numpy())
                else:
                    h = torch.Tensor((self.tau0 * self.previous_embeddings[z].clone() + (1-self.tau0) * h.clone()).detach().numpy())
            current_embeddings[z] = h.clone()
        
        #HADAMARD MLP
        h_src = h[edge_label_index[0]]
        h_dst = h[edge_label_index[1]]
        h_hadamard = torch.mul(h_src, h_dst) #hadamard product
        h = self.postprocess1(h_hadamard)
        h = torch.sum(h.clone(), dim=-1).clone()
        
        #return both 
        #i)the predictions for the current snapshot 
        #ii) the embeddings of current snapshot

        return h, current_embeddings
    
    def loss(self, pred, link_label):
        return self.loss_fn(pred, link_label)    
    
class T3GConvGRU(torch.nn.Module):
    def __init__(self, in_channels, hidden_conv_2):
        super(T3GConvGRU, self).__init__()
        self.gcgru = GConvGRU(in_channels, hidden_conv_2, 2)
        self.post = torch.nn.Linear(hidden_conv_2, 2)
        
        self.loss_fn = BCEWithLogitsLoss()
        
    def reset_parameters(self):
        self.post.reset_parameters()

    def forward(self, x, edge_index, edge_label_index, H=None):
        h = self.gcgru(x, edge_index, H=H)
        hidden = torch.Tensor(h.detach().numpy())
        h = F.relu(h)
        
        #HADAMARD MLP
        h_src = h[edge_label_index[0]]
        h_dst = h[edge_label_index[1]]
        h_hadamard = torch.mul(h_src, h_dst) #hadamard product
        h = self.post(h_hadamard)
        h = torch.sum(h.clone(), dim=-1).clone()
        
        return h, hidden
    
    def loss(self, pred, link_label):
        return self.loss_fn(pred, link_label)
    
class T3EvolveGCNH(torch.nn.Module):
    def __init__(self, num_nodes, in_channels):
        super(T3EvolveGCNH, self).__init__()
        self.evolve = EvolveGCNH(num_nodes, in_channels)
        self.post = torch.nn.Linear(in_channels, 2)
        
        self.loss_fn = BCEWithLogitsLoss()
        
    def reset_parameters(self):
        self.post.reset_parameters()

    def forward(self, x, edge_index, edge_label_index):
        h = self.evolve(x, edge_index)
        h = F.relu(h)
        
        #HADAMARD MLP
        h_src = h[edge_label_index[0]]
        h_dst = h[edge_label_index[1]]
        h_hadamard = torch.mul(h_src, h_dst) #hadamard product
        h = self.post(h_hadamard)
        h = torch.sum(h.clone(), dim=-1).clone()
        
        return h
    
    def loss(self, pred, link_label):
        return self.loss_fn(pred, link_label)
    
class T3EvolveGCNO(torch.nn.Module):
    def __init__(self, in_channels):
        super(T3EvolveGCNO, self).__init__()
        self.evolve = EvolveGCNO(in_channels)
        self.post = torch.nn.Linear(in_channels, 2)
        
        self.loss_fn = BCEWithLogitsLoss()
        
    def reset_parameters(self):
        self.post.reset_parameters()

    def forward(self, x, edge_index, edge_label_index):
        h = self.evolve(x, edge_index)
        h = F.relu(h)
        
        #HADAMARD MLP
        h_src = h[edge_label_index[0]]
        h_dst = h[edge_label_index[1]]
        h_hadamard = torch.mul(h_src, h_dst) #hadamard product
        h = self.post(h_hadamard)
        h = torch.sum(h.clone(), dim=-1).clone()
        
        return h
    
    def loss(self, pred, link_label):
        return self.loss_fn(pred, link_label)