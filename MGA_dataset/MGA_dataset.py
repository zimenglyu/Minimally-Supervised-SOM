import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

class MGA_Data:
    def __init__(self, norm_method="minmax"):
        self.norm_method = norm_method
        print("!!!normalization method is {}".format(self.norm_method))
        if (self.norm_method == "minmax"):
            print("using minmax scaler")
            self.label_scaler = MinMaxScaler()
            self.spectra_scaler = MinMaxScaler()
        elif (self.norm_method == "standard"):
            print("using standard scaler")
            self.label_scaler = StandardScaler()
            self.spectra_scaler = StandardScaler()
        elif (self.norm_method == "robust"):
            print("using robust scaler")
            self.label_scaler = RobustScaler()
            self.spectra_scaler = RobustScaler()
        else:
            print("Error: the scaler is not minmax or standard")
        
    def get_data(self, args):
        # read train data
        input_train_spectra = pd.read_csv(args.train_spectra)
        input_train_spectra = input_train_spectra.drop(columns=['DateTime'])
        input_train_spectra = self.drop_spectra_channels(input_train_spectra, args.remove_first_channels, args.remove_last_channels)
 
        # get number of columns
        self.num_train_columns = len(input_train_spectra.loc[0])

        # read lab result data
        input_lab_spectra = pd.read_csv(args.lab_result_spectra)
        input_lab_labels = pd.read_csv(args.lab_result_label)
        input_lab_spectra = input_lab_spectra.drop(columns=['DateTime'])
        input_lab_labels = input_lab_labels.drop(columns=['DateTime'])
        input_lab_spectra = self.drop_spectra_channels(input_lab_spectra, args.remove_first_channels, args.remove_last_channels)
        self.headers = input_lab_labels.columns
        
        # read testing data
        input_test_spectra = pd.read_csv(args.testing_path)
        input_test_spectra = input_test_spectra.drop(columns=['DateTime'])
        input_test_spectra = self.drop_spectra_channels(input_test_spectra, args.remove_first_channels, args.remove_last_channels)
        
        # get number of coal property columns
        self.num_labels = input_lab_labels.shape[1]
        print("number of coal property columns is {}".format(self.num_labels))
        # get number of samples after merging
        self.num_vali_points = input_lab_spectra.shape[0]
        print("number of validation points is {}".format(self.num_vali_points))
 
        # normalize coal property data
        self.label_scaler.fit(input_lab_labels)
        self.lab_labels = self.label_scaler.transform(input_lab_labels)

        # normalize spectra data
        self.spectra_scaler.fit(input_train_spectra)
        self.train_data = self.spectra_scaler.transform(input_train_spectra)
        self.lab_spectra = self.spectra_scaler.transform(input_lab_spectra)
        self.testing_spectra = self.spectra_scaler.transform(input_test_spectra)

    def drop_spectra_channels(self, data, n_front_channels, n_end_channels):
        if (n_front_channels == 0 and n_end_channels == 0):
            return data
        else:
            print("dropping {} channels from front and {} channels from end".format(n_front_channels, n_end_channels))
            print("data shape before dropping channels is {}".format(data.shape))
            data = data.iloc[:, n_front_channels:]
            if (n_end_channels != 0):
                data = data.iloc[:, :-n_end_channels]
            print("data shape after dropping channels is {}".format(data.shape))
            return data

    def denormalize(self, data):
        print("denormalizing data, data original shape is {}".format(data.shape))
        denorm_data = self.label_scaler.inverse_transform(data)
        denorm_data_df = pd.DataFrame(denorm_data)
        denorm_data_df.columns = self.headers
        return denorm_data_df








