import torch
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, GRUCell, CrossEntropyLoss
from torch_geometric.data import Data
from sklearn.metrics import roc_auc_score,average_precision_score

import random

import copy

import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv,SAGEConv,GATv2Conv, GINConv, Linear

import torch
import networkx as nx
import numpy as np

from torch_geometric_temporal.nn.recurrent import GConvGRU,  EvolveGCNH, EvolveGCNO

class T3ROLAND(torch.nn.Module):
    def __init__(self, input_dim, num_nodes, hidden_conv_1, hidden_conv_2, dropout=0.0, update='mlp', loss=BCEWithLogitsLoss):
        
        super(T3ROLAND, self).__init__()
        #Architecture: 
            #2 MLP layers to preprocess node repr, 
            #2 GCN layer to aggregate node embeddings
            #HadamardMLP as link prediction decoder
        
        #You can change the layer dimensions but 
        #if you change the architecture you need to change the forward method too
        #TODO: make the architecture parameterizable
        
        self.preprocess1 = Linear(input_dim, 256)
        self.preprocess2 = Linear(256, 128)
        self.conv1 = GCNConv(128, hidden_conv_1)
        self.conv2 = GCNConv(hidden_conv_1, hidden_conv_2)
        self.postprocess1 = Linear(hidden_conv_2, 2)
        
        #Initialize the loss function to BCEWithLogitsLoss
        self.loss_fn = loss()

        self.dropout = dropout
        self.update = update
        
        self.tau0 = torch.nn.Parameter(torch.Tensor([0.2]))
        if update=='gru':
            self.gru1 = GRUCell(hidden_conv_1, hidden_conv_1)
            self.gru2 = GRUCell(hidden_conv_2, hidden_conv_2)
        elif update=='mlp':
            self.mlp1 = Linear(hidden_conv_1*2, hidden_conv_1)
            self.mlp2 = Linear(hidden_conv_2*2, hidden_conv_2)
        self.previous_embeddings = None
                                    
        
    def reset_loss(self,loss=BCEWithLogitsLoss):
        self.loss_fn = loss()
        
    def reset_parameters(self):
        self.preprocess1.reset_parameters()
        self.preprocess2.reset_parameters()
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.postprocess1.reset_parameters()
        

    def forward(self, x, edge_index, edge_label_index=None, isnap=0, previous_embeddings=None):
        
        #You do not need all the parameters to be different to None in test phase
        #You can just use the saved previous embeddings and tau
        if previous_embeddings is not None and isnap > 0: #None if test
            self.previous_embeddings = [previous_embeddings[0].clone(),previous_embeddings[1].clone()]
        
        current_embeddings = [torch.Tensor([]),torch.Tensor([])]
        
        #Preprocess node repr
        h = self.preprocess1(x)
        h = h.relu()
        h = torch.Tensor(F.dropout(h, p=self.dropout).detach().numpy())
        h = self.preprocess2(h)
        h = h.relu()
        h = torch.Tensor(F.dropout(h, p=self.dropout).detach().numpy())
        
        #GRAPHCONV
        #GraphConv1
        h = self.conv1(h, edge_index)
        h = h.relu()
        h = torch.Tensor(F.dropout(h, p=self.dropout).detach().numpy())
        #Embedding Update after first layer
        if isnap > 0:
            if self.update=='gru':
                h = torch.Tensor(self.gru1(h, self.previous_embeddings[0].clone()).detach().numpy())
            elif self.update=='mlp':
                hin = torch.cat((h,self.previous_embeddings[0].clone()),dim=1)
                h = torch.Tensor(self.mlp1(hin).detach().numpy())
            else:
                h = torch.Tensor((self.tau0 * self.previous_embeddings[0].clone() + (1-self.tau0) * h.clone()).detach().numpy())
       
        current_embeddings[0] = h.clone()
        #GraphConv2
        h = self.conv2(h, edge_index)
        h = h.relu()
        h = torch.Tensor(F.dropout(h, p=self.dropout).detach().numpy())
        #Embedding Update after second layer
        if isnap > 0:
            if self.update=='gru':
                h = torch.Tensor(self.gru2(h, self.previous_embeddings[1].clone()).detach().numpy())
            elif self.update=='mlp':
                hin = torch.cat((h,self.previous_embeddings[1].clone()),dim=1)
                h = torch.Tensor(self.mlp2(hin).detach().numpy())
            else:
                h = torch.Tensor((self.tau0 * self.previous_embeddings[1].clone() + (1-self.tau0) * h.clone()).detach().numpy())
      
        current_embeddings[1] = h.clone()
        
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