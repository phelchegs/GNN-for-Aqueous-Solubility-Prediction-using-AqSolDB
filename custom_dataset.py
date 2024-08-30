import torch
from torch_geometric.data import Data, Dataset
from deepchem.feat.graph_data import GraphData
import deepchem as dc
import pandas as pd
import os
from tqdm.notebook import tqdm
import rdkit
import logging
logging.getLogger('deepchem').setLevel(logging.ERROR)


class CustomDataset(Dataset):
    def __init__(self, root):
        super(CustomDataset, self).__init__(root)
        self.root = root
        self.graph_list = None

    @property #@property make the function an attribute of the class.
    def raw_file_paths(self):
        return ['curated-solubility-dataset.csv']

    @property
    def processed_file_names(self):
        return ['processed_data.pt']

    def process(self):
        graph_list = []
        mol_converter = dc.feat.MolGraphConvFeaturizer(use_edges = True)
        count = 0
        for raw_path in [os.path.join(self.root, file) for file in self.raw_file_paths]: #loop all data files in the root path. If there are more solubility data, 
            self.data = pd.read_csv(raw_path, header = 0)
            bins, labels = self.classifier() #Classify logS to <-4, [-4,-2], [-2, 0], >0
            self.data['Sol_Label'] = pd.cut(self.data['Solubility'], bins = bins, labels = labels, right = True)
            for index, row in tqdm(self.data.iterrows(), total = self.data.shape[0], desc = "Processing rows"):
                g = mol_converter.featurize(row['SMILES']) #Only use SMILES in this project.
                if type(g[0]) is GraphData:
                    graph = self.convert_pyg(g[0])
                    graph.y = torch.tensor(row['Sol_Label'], dtype = torch.long)
                    graph_list.append(graph)
                    count += 1
        torch.save(graph_list, self.processed_file_names[0])
        print('{} molecules have been processed in file {}.'.format(count, os.path.split(raw_path)[1]))

    def len(self): #Required by torch_geometric.data.Dataset
        if self.graph_list is None:
            self.graph_list = torch.load(self.processed_file_names[0])
        return len(self.graph_list)

    def get(self, idx): #Required by torch_geometric.data.Dataset
        if self.graph_list is None:
            self.graph_list = torch.load(self.processed_file_names[0])
        return self.graph_list[idx]

    def classifier(self):
        return [[float('-inf'), -4, -2, 0, float('inf')], [0, 1, 2, 3]]

    def convert_pyg(self, dc_graph):
        node_features = torch.tensor(dc_graph.node_features, dtype = torch.float)
        edge_index = torch.tensor(dc_graph.edge_index, dtype = torch.long)
        edge_attr = torch.tensor(dc_graph.edge_features, dtype = torch.float)
        return Data(x = node_features, edge_index = edge_index, edge_attr = edge_attr)