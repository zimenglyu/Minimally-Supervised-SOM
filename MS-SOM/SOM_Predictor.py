import sys
sys.path.append('../susi')
from susi import SOMClustering
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from susi.SOMPlots import plot_umatrix, plot_som_histogram
from MGA_dataset.MGA_dataset import MGA_Data
import argparse
import sys
from UMGraph import UMGraph
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import pickle
from collections import Counter

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class SOM_Predictor:
    def __init__(self, args):
        self.n_rows = args.n_rows
        self.n_columns = args.n_columns
        self.n_epochs = args.n_epochs
        self.num_neighbors = args.num_neighbors
        self.dataset_name = args.data_set
        # self.estimate_method = args.estimate_method
        self.norm_method = args.norm_method
        self.output_path = args.output_path
        self.shuffle = args.shuffle_data
        if (self.shuffle):
            self.random_state = 1
            print("shuffling the data")
        else:
            print("not shuffling the data")
            self.random_state = None
        print("random state is {}".format(self.random_state))
        
        print("normalization method is {}".format(self.norm_method))
        print("creating som with {} x {} size".format(self.n_rows, self.n_columns ))
        print("num neighbors {}".format(self.num_neighbors))

        self.mga_data = MGA_Data(self.norm_method)
        self.mga_data.get_data(args)

    # def get_data(self, args):


    def train(self):
        self.som = SOMClustering(n_rows=self.n_rows, n_columns=self.n_columns, n_iter_unsupervised=self.n_epochs, random_state=self.random_state)
        self.som.fit(self.mga_data.train_data)
        print("finished training SOM")
    
    def prepare_for_prediction(self):
        self.u_matrix = self.som.get_u_matrix()
        distance_graph = UMGraph(self.u_matrix)
        self.shortest_distance = distance_graph.get_shortest_distance()
        print("got the shortest distance matrix")
        self.lab_result_cluster = self.som.transform(self.mga_data.lab_spectra)
        mask_count, corr_index = self.create_mask_and_indices(self.lab_result_cluster, (self.n_rows, self.n_columns))
        print("got mask")
        self.find_closest_points(mask_count, corr_index)
        print("got closest points")
        # self.calculate_closest_neighbor_values()
        # print("got look up table")
    
    def make_look_up_table(self, methods):
        if ("weighted" in methods):
            print("Calculating weighted neighbor look up table")
            self.calculate_closest_neighbor_values()
        if ("linear" in methods):
            print("Calculating linear line search look up table")
            self.calculate_lookup_by_line_search()
        if ("poly" in methods):
            print("Calculating poly line search look up table")
            self.calculate_lookup_by_poly_line_search()
        if ("average" in methods):
            print("Calculating average neighbor look up table")
            self.calculate_average_neighbor_values()
        if ("random" in methods):
            print("Calculating random neighbor look up table")
            self.calculate_random_neighbor_values()
        if ("vote" in methods):
            print("Calculating majority vote neighbor look up table")
            self.calculate_vote_neighbor_values()

    def predict(self):
        pass


    def get_random_array(self, m,n):
        a = np.random.rand(m,n)
        return a

    def save_umatrix(self):
        plot_umatrix(self.u_matrix, self.n_rows, self.n_columns)
        if self.lab_result_cluster is not None:
            plt.scatter(self.lab_result_cluster[:,0] *2, self.lab_result_cluster[:, 1]*2, c='r')
            up = []
            medium = []
            down = []
            leftup = []
            leftdown = []
            leftmedium = []
            for i in range(len(self.lab_result_cluster)):
                if [self.lab_result_cluster[i, 0] *2, self.lab_result_cluster[i, 1]*2] not in up:  
                    plt.text(self.lab_result_cluster[i, 0] *2, self.lab_result_cluster[i, 1]*2, str(i+2))
                    up.append([self.lab_result_cluster[i, 0] *2, self.lab_result_cluster[i, 1]*2])
                else:
                    if [self.lab_result_cluster[i, 0] *2, self.lab_result_cluster[i, 1]*2] not in medium:
                        plt.text(self.lab_result_cluster[i, 0] *2, self.lab_result_cluster[i, 1]*2-0.5, str(i+2))
                        medium.append([self.lab_result_cluster[i, 0] *2, self.lab_result_cluster[i, 1]*2])
                    else:
                        if [self.lab_result_cluster[i, 0] *2, self.lab_result_cluster[i, 1]*2] not in down:
                            plt.text(self.lab_result_cluster[i, 0] *2, self.lab_result_cluster[i, 1]*2+0.5, str(i+2))
                            down.append([self.lab_result_cluster[i, 0] *2, self.lab_result_cluster[i, 1]*2])
                        else:
                            if [self.lab_result_cluster[i, 0] *2, self.lab_result_cluster[i, 1]*2] not in leftup:
                                plt.text(self.lab_result_cluster[i, 0] *2-1, self.lab_result_cluster[i, 1]*2, str(i+2))
                                leftup.append([self.lab_result_cluster[i, 0] *2, self.lab_result_cluster[i, 1]*2])
                            else:
                                if [self.lab_result_cluster[i, 0] *2, self.lab_result_cluster[i, 1]*2] not in leftdown:
                                    plt.text(self.lab_result_cluster[i, 0] *2-1, self.lab_result_cluster[i, 1]*2+0.5, str(i+2))
                                    leftdown.append([self.lab_result_cluster[i, 0] *2, self.lab_result_cluster[i, 1]*2])
                                else:
                                    plt.text(self.lab_result_cluster[i, 0] *2-1, self.lab_result_cluster[i, 1]*2-0.5, str(i+2))
                                    leftmedium.append([self.lab_result_cluster[i, 0] *2, self.lab_result_cluster[i, 1]*2])
        plot_title = "U-matrix {}*{}".format( self.n_rows, self.n_columns)
        plt.title(plot_title)
        save_path = os.path.join(args.output_path, "U-matrix_{}_{}_{}_neighbor_{}_{}_{}.png".format(args.data_set, args.n_rows, args.n_epochs, args.num_neighbors, args.norm_method, args.fold))
        plt.savefig(save_path, bbox_inches='tight')
        

    def inverse_number(self, x, fenmu):
        if ( x == 0 ):
            print("this function is called")
            # incase the number x is 0
            return 1/(fenmu)
        else: 
            return 1/x


    def create_mask_and_indices(self, points_2d, shape):
        # Convert the 2D list of points into two 1D arrays of x and y coordinates
        points_array = np.array(points_2d)
        x_coords = points_array[:, 0]
        y_coords = points_array[:, 1]

        # Create a 2D histogram with the same shape as the mask
        H, x_edges, y_edges = np.histogram2d(x_coords, y_coords, bins=[shape[0], shape[1]])

        # The histogram might have floats, but we want ints
        B = H.astype(int)

        # Create an empty 3D list for the indices
        C = [[[] for _ in range(shape[1])] for _ in range(shape[0])]

        # Populate the 3D list with the indices
        for i, point in enumerate(points_2d):
            C[point[0]][point[1]].append(i)

        return B, C

    def create_mask_and_indices(self, points_2d, shape):
        # Convert the 2D list of points into two 1D arrays of x and y coordinates
        points_array = np.array(points_2d)
        x_coords = points_array[:, 0]
        y_coords = points_array[:, 1]

        # Create a 2D histogram with the same shape as the mask
        H, x_edges, y_edges = np.histogram2d(x_coords, y_coords, bins=[shape[0], shape[1]])

        # The histogram might have floats, but we want ints
        B = H.astype(int)

        # Create an empty 3D list for the indices
        C = [[[] for _ in range(shape[1])] for _ in range(shape[0])]

        # Populate the 3D list with the indices
        for i, point in enumerate(points_2d):
            C[point[0]][point[1]].append(i)

        return B, C

    def find_closest_points(self, B, C):
        n = int(np.sqrt(self.shortest_distance.shape[0]))  # Size of one dimension of A and B

        assert self.shortest_distance.shape == (n*n, n*n), "D must have shape (n*n, n*n)"
        assert B.shape == (n, n), "B must have shape (n, n)"

        # Initialize the result array
        self.closest_points = [[[[] for _ in range(self.num_neighbors)] for _ in range(n)] for _ in range(n)]
        self.closest_distances = np.full((n, n, self.num_neighbors), np.inf)

        # Iterate over each cell in A (and each row in D)
        for i in range(n):
            for j in range(n):
                # Create a list of all distances and their corresponding indices
                distances_indices = [(self.shortest_distance[i*n+j][k*n+l], (k, l)) for k in range(n) for l in range(n)]
                # Sort the list by distance
                distances_indices.sort()

                # Initialize a counter for the number of closest points found
                count = 0

                # Iterate over the distances and indices in ascending order of distance
                for distance, (k, l) in distances_indices:
                    if (i == k and j == l):
                        # print("oh shit")
                        continue
                    # If the count at (k, l) in B is non-zero, add the indices from C to the closest points
                    if B[k, l] > 0:
                        for index in C[k][l]:
                            self.closest_points[i][j][count].append(index)
                            self.closest_distances[i, j, count] = distance
                            count += 1

                            # If we have found num_points closest points, break the loop
                            if count == self.num_neighbors:
                                break

                    # If we have found num_points closest points, break the loop
                    if count == self.num_neighbors:
                        break

    def calculate_closest_neighbor_values(self):
        self.weighted_neighbor_table = np.zeros((self.mga_data.num_labels, self.n_rows, self.n_columns))
        fenmu = 0.1
        for prop in range(self.mga_data.num_labels):
            for i in range(self.n_rows):
                for j in range(self.n_columns):
                    distance_sum_w = 0
                    numerator_w = 0
                    for k in range(self.num_neighbors):
                        # part_distance = inverse_number()
                        part_distance = self.inverse_number(self.closest_distances[i,j,k], fenmu)
                        distance_sum_w += part_distance
                        numerator_w += part_distance * self.mga_data.lab_labels[self.closest_points[i][j][k], prop]
                    self.weighted_neighbor_table[prop,i,j] = numerator_w / distance_sum_w
    
    def calculate_average_neighbor_values(self):
        self.average_neighbor_table = np.zeros((self.mga_data.num_labels, self.n_rows, self.n_columns))
        sum_values = 0
        for prop in range(self.mga_data.num_labels):
            for i in range(self.n_rows):
                for j in range(self.n_columns):
                    for k in range(self.num_neighbors):
                        sum_values += self.mga_data.lab_labels[self.closest_points[i][j][k], prop]
                    self.average_neighbor_table[prop,i,j] = sum_values / self.num_neighbors
    
    def calculate_random_neighbor_values(self):
        self.random_neighbor_table = np.zeros((self.mga_data.num_labels, self.n_rows, self.n_columns))
        sum_values = 0
        indices = np.arange(self.mga_data.num_vali_points)
        for prop in range(self.mga_data.num_labels):
            for i in range(self.n_rows):
                for j in range(self.n_columns):
                    np.random.shuffle(indices)
                    for k in range(self.num_neighbors):
                        sum_values += self.mga_data.lab_labels[indices[k], prop]
                    self.random_neighbor_table[prop,i,j] = sum_values / self.num_neighbors
        
    def calculate_lookup_by_line_search(self):
        self.linear_table = np.zeros((self.mga_data.num_labels, self.n_rows, self.n_columns))
        for prop in range(self.mga_data.num_labels):
            for i in range(self.n_rows):
                for j in range(self.n_columns):
                    model = LinearRegression()
                    model.fit(self.closest_distances[i,j].reshape((-1,1)), self.mga_data.lab_labels[self.closest_points[i][j], prop])
                    y_pred = model.predict([[0]])
                    self.linear_table[prop,i,j] = y_pred

    def calculate_lookup_by_poly_line_search(self):
        self.poly_table = np.zeros((self.mga_data.num_labels, self.n_rows, self.n_columns))
        for prop in range(self.mga_data.num_labels):
            for i in range(self.n_rows):
                for j in range(self.n_columns):
                    X = self.closest_distances[i,j].reshape((-1,1))
                    poly = PolynomialFeatures(degree=2)
                    X_poly = poly.fit_transform(X)

                    # Step 3: Create and fit the model
                    model = LinearRegression()
                    model.fit(X_poly, self.mga_data.lab_labels[self.closest_points[i][j], prop])
                    x_new = [[0]]
                    X_new_poly = poly.transform(x_new)
                    y_pred = model.predict(X_new_poly)
                    self.poly_table[prop, i,j] = y_pred
    def calculate_vote_neighbor_values(self):
        self.vote_table = np.zeros((self.mga_data.num_labels, self.n_rows, self.n_columns))
        
        for prop in range(self.mga_data.num_labels):
            for i in range(self.n_rows):
                for j in range(self.n_columns):
                    classes = []
                    for k in range(self.num_neighbors):
                        classes.append(self.mga_data.lab_labels[self.closest_points[i][j][k], prop][0])
                    counter = Counter(classes)
                    # Find the most common element
                    majority_vote = counter.most_common(1)[0][0]
                    self.vote_table[prop,i,j] = majority_vote


    def look_up(self, prediction, method):
        predicted_value = np.zeros((self.mga_data.testing_spectra.shape[0] , self.mga_data.num_labels))
        for pt in range(self.mga_data.testing_spectra.shape[0]):
            for i in range(self.mga_data.num_labels):
                if (method == "weighted"):
                    predicted_value[pt, i] = self.weighted_neighbor_table[i, int(prediction[pt, 0]), int(prediction[pt, 1])]
                elif(method == "linear"):
                    predicted_value[pt, i] = self.linear_table[i, int(prediction[pt, 0]), int(prediction[pt, 1])]
                elif(method == "poly"):
                    predicted_value[pt, i] = self.poly_table[i, int(prediction[pt, 0]), int(prediction[pt, 1])]
                elif(method == "average"):
                    predicted_value[pt, i] = self.average_neighbor_table[i, int(prediction[pt, 0]), int(prediction[pt, 1])]
                elif(method == "random"):
                    predicted_value[pt, i] = self.random_neighbor_table[i, int(prediction[pt, 0]), int(prediction[pt, 1])]
                elif(method == "vote"):
                    predicted_value[pt, i] = self.vote_table[i, int(prediction[pt, 0]), int(prediction[pt, 1])]
        return predicted_value
    
def save_som(predictor, filename):
    
    with open(filename, "wb") as file:
        pickle.dump(predictor, file)

def load_som(filename):
    with open(filename, "rb") as file:
        predictor = pickle.load(file)
    return predictor
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments for SOM")

    parser.add_argument(
        "--train_spectra",
        required=True,
        help="input training dataset with spectra",
    )
    parser.add_argument(
        "--lab_result_spectra", required=True, help="lab result spectra, no labels"
    )
    parser.add_argument(
        "--lab_result_label", required=True, help="lab result labels, no spectra"
    )
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
        type=str2bool,
        help="option to plot and save umatrix plot",
    )
    parser.add_argument(
        "--plot_hist",
        required=False,
        default=False,
        type=str2bool,
        help="option to plot and save SOM hist graph",
    )
    parser.add_argument(
        "--testing_path",
        required=True,
        default="",
        help="make prediction on this file, it can be same as train_spectra file",
    )
    parser.add_argument(
        "--num_neighbors",
        required=False,
        default=5,
        type=int,
        help="number of neighbors used for estimating values",
    )
    parser.add_argument(
        "--save_shortest_distance_csv",
        required=False,
        default=False,
        type=str2bool,
        help="if write the shortest distance path to csv",
    )
    parser.add_argument(
        "--data_set",
        required=True,
        help="data set name, used for saving the results",
    )
    parser.add_argument(
        "--norm_method",
        required=False,
        default="minmax",
        help="normalization method",
    ) 
    parser.add_argument(
        "--estimate_method",
        required=True,
        nargs="+",
        action='append',
        help="a string of methods to estimate values",
    )
    parser.add_argument(
        "--output_path",
        required=True,
        help="path to save prediction",
    )
    parser.add_argument(
        "--fold",
        type=int,
        required=True,
        help="repeat number",
    )
    parser.add_argument(
        "--som_path",
        required=False,
        default="",
        help="path to load som",
    )
    parser.add_argument(
        "--load_som",
        required=False,
        default=False,
        type=str2bool,
        help="option to use saved som",
    )

    parser.add_argument(
        "--remove_first_channels",
        required=False,
        default=0,
        type=int,
        help="number of spectra channels to remove from the beginning",
    )
    parser.add_argument(
        "--remove_last_channels",
        required=False,
        default=0,
        type=int,
        help="number of spectra channels to remove from the end",
    )
    parser.add_argument(
        "--shuffle_data",
        required=False,
        default=False,
        type=str2bool,
        help="option to shuffle the data in SOM training",
    )

    args = parser.parse_args()
    # ------- Train and save the SOM_Predictor -------
    print("--------------------------------------------")
    print("load som is {}".format(args.load_som))
    methods = [item for sublist in args.estimate_method for item in sublist]
    print("Methods used for value estimations are {}".format(methods))
    if (not args.load_som):
        print("No som path provided, training a new som")
        som_predictor = SOM_Predictor(args)
        # som_predictor.get_data(args)
        som_predictor.train()
        print("finished training som")
        som_predictor.prepare_for_prediction()

        som_predictor.make_look_up_table(methods)
        som_name = os.path.join(args.output_path, "{}_{}_{}_neighbor_{}_{}_{}.pickle".format(args.data_set, args.n_rows, args.n_epochs, args.num_neighbors,  args.norm_method, args.fold))
        save_som(som_predictor, som_name)
        print("finished saving som to {}".format(som_name))
    else:
        if (args.som_path == ""):
            print("Error: choose to load SOM but no som path provided")
            sys.exit()
        print("Loading som from {}".format(args.som_path))
        som_predictor = load_som(args.som_path)
        # som_predictor.norm_method = args.norm_method
        print("finished loading som")
        # som_predictor.get_data(args)
    
    # ------- Make umatrix plot -------
    if (args.plot_umatrix):
            som_predictor.save_umatrix()

    # ------- Make prediction -------
    som_predictor.mga_data.get_data(args)
    predicted_cluster = som_predictor.som.transform(som_predictor.mga_data.testing_spectra)
    # for method in ["weighted", "linear", "poly"]:
    for method in methods:
    # for method in ["average", "random", "weighted", "linear", "poly"]:
        prediction = som_predictor.look_up(predicted_cluster, method)
        filename = "{}_{}_{}_{}_neighbor_{}_{}_{}.csv".format(args.data_set, args.n_rows, args.n_epochs, method, args.num_neighbors,  args.norm_method, args.fold)
        filename = os.path.join(args.output_path, filename)
        output_prediction_denormalized = som_predictor.mga_data.denormalize(prediction)
        output_prediction_denormalized.to_csv(filename)
    print("finished making prediction")
    print("--------------------------------------------")
