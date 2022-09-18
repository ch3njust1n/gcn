import numpy as np
import networkx as nx


class GCN(object):
    def __init__(self, graph):
        self.G = graph
        self.num_features = 10
        self.embedding = np.array(self.graph.shape[0], self.num_features)
        self.l0 = GCLayer(self.num_features, 32)
        self.l1 = GCLayer(32, 2)
        self.parameters = [self.l0, self.l1]
        
    
    def __call__(self, x):
        return softmax(self.l1(self.l0(self.G, x)))
    
    
    def backward(self, x):
        self.l0.backward(self.G, self.l1.backward(self.G, x))
        
        
class GCLayer(object):
    def __init__(self, input_dim, output_dim):
        self.weights = np.zeros(input_dim, output_dim)
        self.gradients = None
        
        
    def __call__(self, G, x):
        return relu(G @ x @ self.weights)
    
    
    def backward(self, G, x):
        self.gradients = x
        return relu_derivative(np.transpose(self.weights) @ G @ x)
    
    
    def zero_gradients(self):
        self.gradients = None
    

def relu(x):
    return np.piecewise(x, [x <= 0, x > 0], [0, x])
    
    
def relu_derivative(x):
    return (x > 0) * 1
    
    
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))
    

def renormalization(G):
    A = nx.to_numpy_matrix(G)
    I = np.eye(len(A))
    A_tilde = A + I
    D_tilde = np.zeros(A.shape, int)
    np.fill_diagonal(D_tilde, np.sum(A_tilde, axis=1).flatten())
    D_tilde = np.linalg.inv(D_tilde)
    D_tilde = np.power(D_tilde, 0.5)
    return D_tilde @ A_tilde @ D_tilde


def cross_ent(value):
    pass