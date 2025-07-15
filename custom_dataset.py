import numpy as np
import torch
from torch_geometric.data import Data, Dataset
from deepchem.feat.graph_data import GraphData
import deepchem as dc
import pandas as pd
import os
import tqdm
import logging
logging.getLogger('deepchem').setLevel(logging.ERROR)


class CustomDataset(Dataset):
    def __init__(self, root):
        super(CustomDataset, self).__init__(root)
        self.graph_list = None

    @property
    def raw_file_paths(self):
        return ['curated-solubility-dataset.csv']

    @property
    def processed_file_names(self):
        return ['processed_data.pt']

    def process(self):
        epsilon = 1e-10
        graphs = []
        mol2graph_converter = dc.feat.MolGraphConvFeaturizer(use_edges = True)
        count = 0
        for raw_path in [os.path.join(self.root, file) for file in self.raw_file_paths]:
            data = pd.read_csv(raw_path, header = 0)
            data['LogS'] = np.log10(data['Solubility'] + epsilon)
            smiles_list = data['SMILES'].to_list()
            graph_list = mol2graph_converter.featurize(smiles_list)
            for i, g in tqdm.tqdm(enumerate(graph_list), total = len(graph_list), desc = "Processing graph data converted by DeepChem's MolGraphConvFeaturizer"):
                if isinstance(g, GraphData):
                    graph = self.convert_pyg(g)
                    logS_value = data.iloc[i]['logS']
                    graph.y = torch.tensor([logS_value], dtype=torch.float)
                    graphs.append(graph)
                    count += 1
        torch.save(graphs, self.processed_file_names[0])
        print('{} molecules have been processed in file {}.'.format(count, os.path.split(raw_path)[1]))

    def __len__(self):
        if self.graph_list is None:
            self.graph_list = torch.load(self.processed_file_names[0])
        return len(self.graph_list)

    def __getitem__(self, idx):
        if self.graph_list is None:
            self.graph_list = torch.load(self.processed_file_names[0])
        return self.graph_list[idx]

    def convert_pyg(self, dc_graph):
        node_features = torch.tensor(dc_graph.node_features, dtype = torch.float)
        edge_index = torch.tensor(dc_graph.edge_index, dtype = torch.long)
        edge_attr = torch.tensor(dc_graph.edge_features, dtype = torch.float)
        return Data(x = node_features, edge_index = edge_index, edge_attr = edge_attr)