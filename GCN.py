import datetime
import math
import numpy as np
import torch
from torch import nn, backends
from torch.nn import Module, Parameter
import torch.nn.functional as F
import torch.sparse
from scipy.sparse import coo
import time
from torch_geometric.nn import GatedGraphConv,SAGEConv,GATConv,GCNConv

class GNNModel(nn.Module):
    def __init__(self,n_events,n_user,n_group,hidden_size,device):
        super(GNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.n_user = n_user
        self.n_event = n_events
        self.n_group = n_group
        self.device = device
        self.group_embedding = nn.Embedding(self.n_group, self.hidden_size)
        self.event_embedding = nn.Embedding(self.n_event, self.hidden_size)
        self.user_embedding = nn.Embedding(self.n_user, self.hidden_size)

        self.userconv1 = GCNConv(self.hidden_size,self.hidden_size) #GCN
        self.userconv2 = GCNConv(self.hidden_size,self.hidden_size)
        self.groupconv1 = GCNConv(self.hidden_size, self.hidden_size)
        self.groupconv2 = GCNConv(self.hidden_size,self.hidden_size)
        self.eventnet = nn.Linear(self.hidden_size,self.hidden_size)

        self.loss_function = nn.MSELoss()

    def forward(self, data):
        x, user_edge_index, user_edge_weight, group_edge_index, group_edge_weight = \
            data.x, data.user_edge_index, data.user_edge_weight, data.group_edge_index, data.group_edge_weight

        user_emb = F.relu(self.userconv1(self.user_embedding.weight,user_edge_index,user_edge_weight)) # GCN
#        user_emb = F.relu(self.userconv1(self.user_embedding.weight,user_edge_index))#,user_edge_weight)) Attention
        user_emb = self.userconv2(user_emb,user_edge_index,user_edge_weight)
        group_emb = F.relu(self.groupconv1(self.group_embedding.weight,group_edge_index,group_edge_weight))

        return user_emb, group_emb




