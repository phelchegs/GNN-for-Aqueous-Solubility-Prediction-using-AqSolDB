import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.data import Data

class CustomGCNConv(MessagePassing):

    def __init__(self, node_dim, edge_dim, inside_dim):
        super(CustomGCNConv, self).__init__(aggr = 'add') #How the messages are treated when they aggregate to one node.
        self.lin = nn.Linear(node_dim, inside_dim) #Node (atom) tensor dim should be converted to inside dim by linear trans.
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim, 32),
            nn.ReLU(),
            nn.Linear(32, inside_dim)
        ) #Edge (bond) tensor dim should be converted to inside dim by a linear trans (edge dim to 32), relu, and another linear (32 to inside dim).

    def forward(self, x, edge_index, edge_attr): 
        
        #Here, x is the node tensor with shape [num of nodes, node dim]. Edge_index is the edge tensor with shape [2, num of edges], where the last dim shows the connection between nodes. 
        #Edge_attr is the edge tensor with shape [num of edges, num of edge features].
        
        if edge_attr.dim() != 2 or edge_attr.size(0) != edge_index.size(1):
            raise ValueError('Edge attribute must be of shape [column of edge_index, edge_attr_dim]')
        x = self.lin(x)
        return self.propagate(edge_index = edge_index, x = x, edge_attr = edge_attr) #Magic function that combines edge index, attribute, and nodes.
        #It requires aggr, message, and update.

    def message(self, x_j, edge_attr): #compute message for each edge.
        edge_attr = self.edge_mlp(edge_attr)
        return x_j + edge_attr

    def update(self, x): #The aggregated message is used to updated the node's feature.
        return x 

class CustomGCN(torch.nn.Module):

    def __init__(self, node_dim, edge_dim, inside_dim, dropout = 0.2): #The more convolution layers, the better. I choose 3 because of the limited training data.
        super(CustomGCN, self).__init__()
        self.conv1 = CustomGCNConv(node_dim = node_dim, edge_dim = edge_dim, inside_dim = inside_dim)
        self.conv2 = CustomGCNConv(node_dim = inside_dim, edge_dim = edge_dim, inside_dim = 32)
        self.conv3 = CustomGCNConv(node_dim = 32, edge_dim = edge_dim, inside_dim = 16)
        self.dropout = dropout
        self.lin = nn.Linear(16, 4) #I need 4 logits output to fit for 4 classes of the solubility.

    def forward(self, data, batch): #Data is the dataloader.
        x, edge_index, edge_attr, y = data.x, data.edge_index, data.edge_attr, data.y
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x = self.conv3(x, edge_index, edge_attr)
        x = global_mean_pool(x, batch) #Batch vector indicating graph membership, the output of global_mean_pool is a pooled representation of each entire graph in a batch.
        x = F.dropout(x, p = self.dropout)
        logit = self.lin(x)
        output = F.softmax(logit, dim = 1)
        loss = nn.CrossEntropyLoss()(logit, data.y.long()) #CELoss for multiclass classification.
        return output, loss