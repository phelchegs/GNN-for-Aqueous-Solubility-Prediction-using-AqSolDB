import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.data import Data

class CustomGCNConv(MessagePassing):

    def __init__(self, node_dim, edge_dim, inside_dim):
        super(CustomGCNConv, self).__init__(aggr = 'add')
        self.lin = nn.Linear(node_dim, inside_dim)
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim, 32),
            nn.ReLU(),
            nn.Linear(32, inside_dim)
        )

    def forward(self, x, edge_index, edge_attr):
        if edge_attr.dim() != 2 or edge_attr.size(0) != edge_index.size(1):
            raise ValueError('Edge attribute must be of shape [column of edge_index, edge_attr_dim]')
        x = self.lin(x)
        return self.propagate(edge_index = edge_index, x = x, edge_attr = edge_attr)

    def message(self, x_j, edge_attr):
        edge_attr = self.edge_mlp(edge_attr)
        return x_j + edge_attr

    def update(self, x):
        return x

class CustomGCN(torch.nn.Module):

    def __init__(self, node_dim, edge_dim, inside_dim, dropout = 0.2):
        super(CustomGCN, self).__init__()
        self.conv1 = CustomGCNConv(node_dim = node_dim, edge_dim = edge_dim, inside_dim = inside_dim)
        self.conv2 = CustomGCNConv(node_dim = inside_dim, edge_dim = edge_dim, inside_dim = 32)
        self.conv3 = CustomGCNConv(node_dim = 32, edge_dim = edge_dim, inside_dim = 16)
        self.dropout = dropout
        self.lin = nn.Linear(16, 4)

    def forward(self, data, batch):
        x, edge_index, edge_attr, y = data.x, data.edge_index, data.edge_attr, data.y
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x = self.conv3(x, edge_index, edge_attr)
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p = self.dropout)
        logit = self.lin(x)
        output = F.softmax(logit, dim = 1)
        loss = nn.CrossEntropyLoss()(logit, y.long())
        return output, loss