# from os import umask
from susi import SOMClustering
# import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# import pandas as pd
# from susi.SOMPlots import plot_umatrix
from MGA_dataset.MGA_dataset import MGA_Data
import argparse
# from typing import List, Tuple
import matplotlib
import sys
from shortest_path import Graph, dijkstra

from sklearn.metrics import mean_squared_error

def find_shortest_path( bum1, bum2, distance_graph, n_col):
    m = abs(bum1[0] - bum2[0])
    n = abs(bum1[1] - bum2[1])
    if m == 0 and n == 0:
        return 0
    else:
        start = get_graph_index(bum1[0], bum1[1], n_col)
        end = get_graph_index(bum2[0], bum2[1], n_col)
        distance_graph.reset()
        D = dijkstra(distance_graph, start)
        return D[end]    
    
def get_graph_index(i,j,row_size):
    graph_i = i*row_size + j
    return graph_i

# generate random array, float between [0, 1]
def get_random_array(m,n):
    a = np.random.rand(m,n)
    return a
    
def shortest_path_matrix(bum_list, u_matrix, n_rows, n_columns):
    num_points = len(bum_list)
    shortest_distance = np.zeros((num_points, num_points)) - 1
    distance_graph = make_graph(u_matrix, n_rows,n_columns)
    for i in range(num_points):
        for j in range(num_points):
            bum1 = bum_list[i]
            bum2 = bum_list[j]
            shortest_distance[i,j] = find_shortest_path(bum1, bum2,distance_graph, n_columns)
    return shortest_distance

def make_graph(umatrix,m,n):
    distance_graph = Graph(m*n)
    d_m = len(u_matrix)-1

    for i in range(m):
        for j in range(n):
            graph_i = get_graph_index(i,j, n)
            if (i + 1 < m):
                graph_i_up = get_graph_index(i+1,j, n)
                distance_graph.add_edge(graph_i, graph_i_up, umatrix[d_m-(i*2+1),j*2])
            if (j + 1 < n): 
                graph_i_right = get_graph_index(i,j+1, n)
                distance_graph.add_edge(graph_i, graph_i_right, umatrix[d_m-(i*2),j*2+1])
    return distance_graph

def inverse_number(x, fenmu):
    if ( x == 0 ):
        # incase the number x is 0
        return 1/(fenmu)
    else: 
        return 1/x

def estimate_value(shortest_distance, num_neighbors,  num_properties, fenmu):
    weighted_neighbor_pred = np.zeros((1, num_properties))
    num_neighbors = int(num_neighbors)
    # for i in range(num_vali_points):
    close_index = np.argsort(shortest_distance[0])
    close_index = close_index[close_index != 0]
    close_index = close_index[:num_neighbors]
    # print(close_index)
    for k in range(num_properties):
        distance_sum_w = 0
        numerator_w = 0
        line_count = 0
        for j in close_index:
            part_distance = inverse_number(shortest_distance[0,j], fenmu)
            distance_sum_w += part_distance
            numerator_w += part_distance * CT_data.test_ground_truth[j-1, k]
            line_count += 1
        weighted_neighbor_pred[0,k] = numerator_w / distance_sum_w  
    return weighted_neighbor_pred

def validate_model(shortest_distance, num_neighbors, num_vali_points, num_properties, fenmu):
    weighted_neighbor_pred = np.zeros((num_vali_points, num_properties))
    average_neighbor_pred = np.zeros((num_vali_points, num_properties))
    random_neighbor_pred = np.zeros((num_vali_points, num_properties))
    num_neighbors = int(num_neighbors)
    for i in range(num_vali_points):
        close_index = np.argsort(shortest_distance[i])
        random_index = np.copy(close_index)
        np.random.shuffle(random_index)
        close_index = close_index[:num_neighbors]
        random_index = random_index[:num_neighbors]
        for k in range(num_properties):
            distance_sum_w = 0
            numerator_w = 0
            numerator_a = 0
            line_count = 0
            for j in close_index:
                part_distance = inverse_number(shortest_distance[i,j], fenmu)
                distance_sum_w += part_distance
                numerator_w += part_distance * CT_data.test_ground_truth[j, k]
                numerator_a += CT_data.test_ground_truth[j, k]
                line_count += 1
            average_neighbor_pred[i,k] = numerator_a / line_count
            weighted_neighbor_pred[i,k] = numerator_w / distance_sum_w
            numerator_r = 0
            line_count = 0
            for j in random_index:               
                numerator_r += CT_data.test_ground_truth[j, k]
                line_count += 1
            random_neighbor_pred[i,k] = numerator_r / line_count
    return weighted_neighbor_pred, average_neighbor_pred, random_neighbor_pred

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments for SOM")

    parser.add_argument(
        "--train_spectra",
        required=True,
        help="input training dataset with spectra",
    )
    parser.add_argument(
        "--test_spectra", required=True, help="test dataset with spectra"
    )
    parser.add_argument(
        "--test_lab_result", required=True, help="test data with lab results"
    )

    # parser.add_argument('--ct_train', required=False, default=False, help='path to coal tracker train data')
    # parser.add_argument('--ct_test', required=False, default=False, help='path to coal tracker test data')
    parser.add_argument(
        "--n_columns",
        required=True,
        type=int,
        help="number of columns for SOM",
    )
    parser.add_argument(
        "--n_rows", required=True, type=int, help="number of rows for SOM"
    )
    parser.add_argument(
        "--n_epochs",
        required=True,
        type=int,
        help="number of epochs for SOM training",
    )
    parser.add_argument(
        "--plot_umatrix",
        required=False,
        default=False,
        type=bool,
        help="option to plot and save umatrix plot",
    )
    parser.add_argument(
        "--plot_hist",
        required=False,
        default=False,
        type=bool,
        help="option to plot and save SOM hist graph",
    )
    parser.add_argument(
        "--do_prediction",
        required=False,
        default="",
        help="if use the code for prediction, if it contains path to test file, then do testing",
    )
    parser.add_argument(
        "--num_neighbors",
        required=False,
        default=5,
        help="number of neighbors used for estimating values",
    )
    parser.add_argument(
        "--save_shortest_distance_csv",
        required=False,
        default=False,
        help="if write the shortest distance path to csv",
    )
    parser.add_argument(
        "--data_set",
        required=True,
        help="data set name, used for saving the results",
    )

    args = parser.parse_args()
    n_columns = args.n_columns
    n_rows = args.n_rows
    num_neighbors = args.num_neighbors
    dataset_name = args.data_set
    if not args.do_prediction:
        do_prediction = False
        prediction_path = ""
    else:
        do_prediction = True
        prediction_path = args.do_prediction
    print("do prediction {}".format(do_prediction))
    print("creating som with {} x {} size".format(n_columns, n_rows))
    print("num neighbors {}".format(num_neighbors))

    CT_data = MGA_Data("minmax", prediction_path)
    CT_data.get_data(args.train_spectra, args.test_spectra, args.test_lab_result)

    # train SOM 
    som = SOMClustering(n_rows=n_rows, n_columns=n_columns, n_iter_unsupervised=args.n_epochs)
    som.fit(CT_data.train_data)

    # get u_matrix
    u_matrix = som.get_u_matrix()

    # make prediction with SOM
    if (do_prediction):
        prediction_lab = som.transform(CT_data.test_data)
        prediction_train = som.transform(CT_data.test_data_prediction)

        # prediction = som.transform(CT_data.test_data_prediction)
    else:
        prediction = som.transform(CT_data.test_data)
    # get shortest path
    num_properties = CT_data.num_CT

    if (do_prediction):
        print("num properties is {}, use {} neighbors".format(num_properties, num_neighbors))
        value_estimates = []
        final_prediction = []
        fenmu = 0.1
        count = 0
        for dp in prediction_train:
            if (count % 1 == 0):
                print("count is {}".format(count))
            for i in range(2):
                prediction = np.vstack([dp, prediction_lab])
                shortest_distance = shortest_path_matrix(prediction, u_matrix, n_rows, n_columns)
                weighted_neighbor_pred = estimate_value(shortest_distance, num_neighbors, num_properties, fenmu)
                value_estimates.append(weighted_neighbor_pred)
            estimation = np.mean(value_estimates, axis=0)
            final_prediction.append(estimation)
            count += 1
        final_prediction = np.squeeze(final_prediction)
        print(np.array(final_prediction).shape)
        output_prediction_denormalized = CT_data.denormalize(final_prediction)
        output_prediction_denormalized.to_csv(dataset_name + "_prediction.csv")
    else:
        shortest_distance = shortest_path_matrix(prediction, u_matrix, n_rows, n_columns)
        print("num properties is {}, use {} neighbors".format(num_properties, num_neighbors))
        weighted_acc = []
        avg_acc = []
        random_neighbor_acc = []
        random_acc = []
        final_prediction = []
        fenmu = 0.1

        for i in range(20):
            
            weighted_neighbor_pred, average_neighbor_pred, random_neighbor_pred = validate_model(shortest_distance, num_neighbors, CT_data.num_vali_points, num_properties, fenmu)
            weighted_som_acc = mean_squared_error(weighted_neighbor_pred, CT_data.test_ground_truth[:CT_data.num_vali_points])
            average_som_acc = mean_squared_error(average_neighbor_pred, CT_data.test_ground_truth[:CT_data.num_vali_points])
            random_som_acc = mean_squared_error(random_neighbor_pred, CT_data.test_ground_truth[:CT_data.num_vali_points])

            random_guess_array = get_random_array(CT_data.num_vali_points, num_properties)
            random_guess_acc = mean_squared_error(random_guess_array, CT_data.test_ground_truth[:CT_data.num_vali_points])
            weighted_acc.append(weighted_som_acc)
            avg_acc.append(average_som_acc)
            random_neighbor_acc.append(random_som_acc)
            random_acc.append(random_guess_acc)
            final_prediction.append(weighted_neighbor_pred)
        # print("inverse number is {}".format(fenmu))
        
        print("10 repeat average using {} neighbors: ".format(num_neighbors))
        print("random guessing acc is %.4f" % (np.mean(random_acc)))
        print("weighted neighbor is %.4f" %(np.mean(weighted_acc)))
        print("average_neighbor is %.4f" % (np.mean(avg_acc)))
        print("random neighbor acc is %.4f" % (np.mean(random_neighbor_acc)))
        print()
        
        output_prediction = np.mean(final_prediction, axis=0)
        output_prediction_denormalized = CT_data.denormalize(output_prediction)
        output_prediction_denormalized.to_csv(dataset_name + "_lab_prediction.csv")