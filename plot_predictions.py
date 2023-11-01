# plot code to compare predicted vs actual values
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler

readable_names = {
        "AshContent": "Ash, wt.% as-fired",
        "BaseAcidRatio": "B\A Ratio",
        "H2OContent": "Moisture, wt.% as-fired",
        "BTU": "Heating Value, BTU\lb as fired",
        "SiContent": "Silicon, wt.% in ash",
        "AlContent": "Aluminum, wt.% in ash",
        "FeContent": "Iron, wt.% in ash",
        "CaContent": "Calcium, wt.% in ash",
        "MgContent": "Magnesium, wt.% in ash",
        "KContent": "Potassium, wt.% in ash",
        "NaContent": "Sodium, wt.% in ash",
        "SOContent": "Sulfur trioxide, wt.% in ash",
        "SContent": "Sulfur, wt.% in ash",
        "TiContent": "Titanium, wt.% in ash"
    }

# def plot_predictions(y_pred, y_test, y_ct, datetime, title):
#     fig, ax = plt.subplots(1,1, figsize=(20, 10 ))
#     x = np.arange(len(y_pred))
#     ax.plot(x, y_pred, label='Predicted')
#     ax.plot(x, y_test, label='True Label')
#     ax.plot(x, y_ct, label='CT')
#     # plt.gcf().autofmt_xdate()
#     ax.set_xticks(x)
#     ax.set_xticklabels([datetime[i] for i in x])
#     ax.set_xticklabels(datetime, rotation=90, ha='right')
#     plt.xticks(fontsize=15)
#     plt.yticks(fontsize=15)
#     plt.legend(fontsize = 17)
#     # plt.xlabel('Timestamp')
#     plt.title(title, fontsize = 20)
#     plt.tight_layout()
#     plt.savefig(title + '.png')

def plot_predictions(y_som, y_label, y_regression, data_name, para_name):
    num_train = 80 
    fig, ax = plt.subplots(1,1, figsize=(20, 10 ))
    x = np.arange(len(y_som))
    ax.plot(x, y_som, label='SOM Predictions')
    ax.plot(x, y_label, label='Label')
    ax.plot(x[num_train:], y_regression[num_train:], label='GPR RBF')
    # ax.set_xticks(x)
    # ax.set_xticklabels([ids[i] for i in x])
    # ax.set_xticklabels(ids, rotation=90, ha='right')
    # plt.xticks(fontsize=20)
    # plt.yticks(fontsize=20)
    plt.legend(fontsize = 20)
    plt.xlabel('Sample', fontsize = 20)
    plt.ylabel(para_name, fontsize = 20)
    plt.title(data_name + para_name, fontsize = 20)
    plt.tight_layout()
    plt.savefig(data_name + para_name + '.png')

# main function
if __name__ == '__main__':
    # label_path = "/Users/zimenglyu/Documents/datasets/microbeam/PPM/combined/cyclone_10/Cyclone_10_202303_202105_202209_lab_results.csv"
    # som_path = "/Users/zimenglyu/Documents/code/git/susi/SOM_cyclone_10_no_shuffle/combined/30/combined_30_20000_weighted_neighbor_7_standard_0.csv"
    # regression_pred_path = "/Users/zimenglyu/Documents/code/git/dataset_toolbox/regression/results_NEW/0.8/gaussian/combined_Matern_standard_4.csv"
    label_path = "/Users/zimenglyu/Documents/datasets/regression/energydata_100_label.csv"
    som_path = "/Users/zimenglyu/Documents/code/git/susi/SOM_energy/energy/30/energy_30_20000_weighted_neighbor_10_standard_0.csv"
    regression_pred_path = "/Users/zimenglyu/Documents/code/git/dataset_toolbox/regression/results_energy/0.8/gaussian/combined_RBF_standard_3.csv"

    
    y_label = pd.read_csv(label_path)
    y_som = pd.read_csv(som_path)
    y_regression = pd.read_csv(regression_pred_path, usecols=[1,2])
    print(y_regression)
    # y_ct = pd.read_csv(ct_path)
    y_label['DateTime']= pd.to_datetime(y_label['DateTime'])
    datetime_string = y_label['DateTime']
    # MTI_ID = y_test['MTI_ID']
    # print(datetime_string)
    # y_test.set_index('DateTime', inplace=True)
    headers = y_som.columns
    # print(headers)
    # plot predictions
    n = y_som.shape[1]
    num_rows = min(y_som.shape[0], y_som.shape[0])

    for i in range(1, n):
        para_name = headers[i].split("_")[0]
        plot_predictions(y_som.iloc[:num_rows,i], y_label.iloc[:num_rows,i], y_regression.iloc[:num_rows,i], "Energy -- RBF ", para_name)
