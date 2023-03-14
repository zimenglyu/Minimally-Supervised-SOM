from queue import PriorityQueue
import math
import pandas as pd
import numpy as np
class Graph:
    def __init__(self, num_of_vertices):
        self.v = num_of_vertices
        self.edges = [[-1 for i in range(num_of_vertices)] for j in range(num_of_vertices)]
        self.visited = []
    def add_edge(self, u, v, weight):
        self.edges[u][v] = weight
        self.edges[v][u] = weight
    def reset(self):
        self.visited = []
    def print_edge(self):
        edge = pd.DataFrame(np.squeeze(np.array(self.edges, dtype=object)))
        # DF_U = pd.DataFrame(np.squeeze(u_matrix))
        # name = "{}_umatrix_{}_{}_e{}".format(args.dataset_name, n_rows, n_columns, args.n_epochs)
        edge.to_csv("edges" + ".csv")

def dijkstra(graph, start_vertex):
    D = {v:float('inf') for v in range(graph.v)}
    D[start_vertex] = 0

    pq = PriorityQueue()
    pq.put((0, start_vertex))

    while not pq.empty():
        (dist, current_vertex) = pq.get()
        graph.visited.append(current_vertex)

        for neighbor in range(graph.v):
            if graph.edges[current_vertex][neighbor] != -1:
                distance = graph.edges[current_vertex][neighbor]
                if neighbor not in graph.visited:
                    old_cost = D[neighbor]
                    new_cost = D[current_vertex] + distance  * distance
                    if new_cost < old_cost:
                        pq.put((new_cost, neighbor))
                        D[neighbor] = new_cost
    for i in range(len(D)):
        if (D[i] != 0):
            D[i] = math.sqrt(D[i])
    return D
