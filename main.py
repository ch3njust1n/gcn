'''

'''

import config
import argparse
import networkx as nx
import numpy as np
from pyvis.network import Network
from model import GCN, renormalization, cross_ent
from optimizer import GradientDescent


def visualiza_graph(G):
    graphvis = Network(notebook=True)
    graphvis.from_nx(G)
    graphvis.show('karate.html')
    
    
def train(model, loss, epochs, G, dataset, opt):
    
    for e in range(epochs):
        epoch_loss = 0
        for x, y in dataset:
            epoch_loss += loss(model(x), y)
            
        model.backward()
        opt.step()
        
        epoch_loss /= len(dataset)
        print(f'epoch {e} loss: {epoch_loss}')
        
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, help='Configuration file')
    args = parser.parse_args()
    epochs = 1
    lr = 1
    dataset = []
    
    # cfg = config.Configuration(args.config)
    with open('data/soc-karate/soc-karate.mtx', 'r') as file:
        G = nx.read_edgelist(file)
        A_hat = renormalization(G)

        model = GCN(A_hat)
        features = np.eye(G.number_of_nodes())
        opt = GradientDescent(model.parameters, lr)
        train(model, cross_ent, epochs, G, features, opt)
    

if __name__ == '__main__':
    main()