import numpy as np
import networkx as nx


class GraphConv(object):
    def __init__(self, width):
        self.weights = np.array()


class GraphConv(object):
    def __init__(self, graph, layers=1):
        self.graph = graph
        self.height = self.graph.shape[0]
        self.width = self.graph.shape[1]
        self.layers = layers
        self.hidden_layers = [GraphConv() for _ in range(self.layers)]
        
    
    def forward(self, x):
        pass


def laplacian(G):
    A = nx.to_numpy_matrix(G)
    I = np.eye(A.shape)
    A_hat = A - I
    