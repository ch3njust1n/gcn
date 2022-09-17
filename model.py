import numpy as np
import networkx as nx


class GraphConv(object):
    def __init__(self, width):
        self.weights = np.array()


class GCN(object):
    def __init__(self, graph, layers=1):
        self.graph = graph
        self.height = self.graph.shape[0]
        self.width = self.graph.shape[1]
        self.layers = layers
        self.hidden_layers = [GraphConv() for _ in range(self.layers)]
        self.gradients = []
        
    
    def forward(self, x):
        for weights in self.layers:
            x = relu(np.matmul(np.matmul(self.graph, x), weights))
        return x
    
    
    def backward(self):
        pass
        


def relu(x):
    np.piecewise(x, [x <= 0, x > 0], [0, x])
    
    
def reli_derivative(x):
    return (x > 0) * 1


def renormalization(G):
    A = nx.to_numpy_matrix(G)
    I = np.eye(len(A))
    A_tilde = A + I
    D_tilde = np.zeros(A.shape, int)
    np.fill_diagonal(D_tilde, np.sum(A_tilde, axis=1).flatten())
    D_tilde = np.linalg.inv(D_tilde)
    D_tilde = np.power(D_tilde, 0.5)
    return np.matmul(np.matmul(D_tilde, A_tilde), D_tilde)