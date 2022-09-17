'''

'''

import config
import argparse
import networkx as nx
import numpy as np
from pyvis.network import Network

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, help='Configuration file')
    args = parser.parse_args()
    
    # cfg = config.Configuration(args.config)
    
    with open('data/soc-karate/soc-karate.mtx', 'r') as file:
        G = nx.read_edgelist(file)
        A = nx.to_numpy_matrix(G)
        
        graphvis = Network(notebook=True)
        graphvis.from_nx(G)
        graphvis.show('karate.html')


if __name__ == '__main__':
    main()