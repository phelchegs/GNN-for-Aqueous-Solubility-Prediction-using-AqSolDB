import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
# from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops

class CustomGCNConv(MessagePassing):

    def __init__(self, node_dim, edge_dim, inside_dim):
        super(CustomGCNConv, self).__init__(aggr = 'add')
        self.lin = nn.Linear(node_dim, inside_dim)
        self.edge_mlp = nn.Linear(edge_dim, inside_dim)

    def forward(self, x, edge_index, edge_attr):
        if edge_attr.dim() != 2 or edge_attr.size(0) != edge_index.size(1):
            raise ValueError('Edge attribute must be of shape [column of edge_index, edge_attr_dim]')
        self_loops_edge_idx, self_loops_edge_attr = add_self_loops(edge_index, edge_attr = edge_attr, fill_value = 0.0, num_nodes = x.size(0))
        x = self.lin(x)
        return self.propagate(edge_index = self_loops_edge_idx, x = x, edge_attr = self_loops_edge_attr)

    def message(self, x_j, edge_attr):
        edge_attr = self.edge_mlp(edge_attr)
        return x_j + edge_attr

    def update(self, x):
        return x

class CustomGCN(torch.nn.Module):

    def __init__(self, node_dim, edge_dim, hidden_dim, dropout = 0.2, training = True):
        super(CustomGCN, self).__init__()
        self.conv1 = CustomGCNConv(node_dim = node_dim, edge_dim = edge_dim, inside_dim = hidden_dim)
        self.conv2 = CustomGCNConv(node_dim = hidden_dim, edge_dim = edge_dim, inside_dim = hidden_dim)
        self.conv3 = CustomGCNConv(node_dim = hidden_dim, edge_dim = edge_dim, inside_dim = hidden_dim)
        self.norm = nn.BatchNorm1d(hidden_dim)
        self.dropout = dropout
        self.lin = nn.Linear(hidden_dim, 1)
        self.training = training

    def forward(self, data, batch):
        x, edge_index, edge_attr, y = data.x, data.edge_index, data.edge_attr, data.y
        x1 = F.dropout(F.relu(self.norm(self.conv1(x, edge_index, edge_attr))), p = self.dropout, training = self.training)
        x2 = F.dropout(F.relu(self.norm(self.conv2(x1, edge_index, edge_attr))), p = self.dropout, training = self.training)
        x2 = x1 + x2
        x3 = F.relu(self.norm(self.conv3(x2, edge_index, edge_attr)))
        x3 = x3 + x2
        x_out = global_mean_pool(x3, batch)
        output = self.lin(x_out)
        loss = F.mse_loss(output.view(-1), y.long().view(-1))
        return output, loss if self.training else output