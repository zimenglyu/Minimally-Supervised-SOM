import numpy as np
import networkx as nx
import pandas as pd

class UMGraph:
    def __init__(self, umatrix):
        self.rows = int((umatrix.shape[0] + 1) / 2)
        self.cols = int((umatrix.shape[1] + 1) / 2)
        print("building graph from umatrix size of:")
        print(self.rows, self.cols)
        # self.rows, self.cols = umatrix.shape
        self.G = nx.Graph()
        d_m = umatrix.shape[0] - 1

        # Create nodes
        for i in range(self.rows):
            for j in range(self.cols):
                self.G.add_node((i, j))

        # Create weighted edges
        for i in range(self.rows):
            for j in range(self.cols):
                if i < self.rows - 1:
                    self.G.add_edge((i, j), (i+1, j), weight=umatrix[d_m-(2*i+1)][2*j])
                if j < self.cols - 1:
                    self.G.add_edge((i, j), (i, j+1), weight=umatrix[d_m-(2*i)][2*j+1])

        self.num_cells = self.rows * self.cols

        # Calculate shortest paths and create distance matrix
        self.distance_matrix = np.zeros((self.num_cells, self.num_cells))

        for i, node1 in enumerate(self.G.nodes):
            for j, node2 in enumerate(self.G.nodes):
                self.distance_matrix[i][j] = nx.dijkstra_path_length(self.G, node1, node2)
        data = pd.DataFrame(self.distance_matrix)
        data.to_csv("distance_matrix_nx.csv")
    
    def shortest_distance_between(self, cell1, cell2):
        # Convert cell coordinates to node indices
        node1 = cell1[0] * self.cols + cell1[1]
        node2 = cell2[0] * self.cols + cell2[1]
        return self.distance_matrix[node1][node2]
    
    def get_shortest_distance(self):
        return self.distance_matrix
    # def shortest_distance(self, cell1):
    #     # Convert cell coordinates to node indices
    #     return nx.single_source_dijkstra_path_length(self.G, cell1, weight='weight')
