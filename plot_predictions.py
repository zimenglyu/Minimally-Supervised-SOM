# plot code to compare predicted vs actual values
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler

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

def plot_predictions(y_pred, y_test, datetime, title):
    fig, ax = plt.subplots(1,1, figsize=(20, 10 ))
    x = np.arange(len(y_pred))
    ax.plot(x, y_pred, label='Predicted')
    ax.plot(x, y_test, label='True Label')
    # ax.plot(x, y_ct, label='CT')
    # plt.gcf().autofmt_xdate()
    # ax.set_xticks(x)
    # ax.set_xticklabels([datetime[i] for i in x])
    # ax.set_xticklabels(datetime, rotation=90, ha='right')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(fontsize = 17)
    # plt.xlabel('Timestamp')
    plt.title(title, fontsize = 20)
    plt.tight_layout()
    plt.savefig(title + '.png')

# main function
if __name__ == '__main__':

    pred_path = "/Users/zimenglyu/Documents/code/git/susi/SOM_cyclone_3/combined/30/combined_30_20000_weighted_neighbor_7_standard_0.csv"
    test_path = "/Users/zimenglyu/Documents/datasets/microbeam/PPM/combined/cyclone_3/Cyclone_3_combined_lab_results.csv"
    # ct_path = "/Users/zimenglyu/Documents/datasets/microbeam/PPM/combined/Cyclone_10_202303_202105_202209_lab_results_ct.csv"

        # test_path = "/Users/zimenglyu/Documents/datasets/microbeam/PPM/2022-09/MGA_10_2022_09_clean.csv"
    # pred_path = "/Users/zimenglyu/Documents/code/git/susi/SOM_cyclone_10/combined/30/combined_30_20000_weighted_neighbor_7_standard_0.csv"
    # test_path = "/Users/zimenglyu/Documents/datasets/microbeam/PPM/combined/cyclone_10/Cyclone_10_202303_202105_202209_lab_results.csv"
    # ct_path = "/Users/zimenglyu/Documents/datasets/microbeam/PPM/combined/Cyclone_10_202303_202105_202209_ct.csv"

    
    y_pred = pd.read_csv(pred_path)
    y_test = pd.read_csv(test_path)
    # y_ct = pd.read_csv(ct_path)
    y_test['DateTime']= pd.to_datetime(y_test['DateTime'])
    datetime_string = y_test['DateTime']
    # print(datetime_string)
    # y_test.set_index('DateTime', inplace=True)
    headers = y_pred.columns
    # print(headers)
    # plot predictions
    n = y_pred.shape[1]
    num_rows = min(y_pred.shape[0], y_test.shape[0])
    print(num_rows)
    for i in range(1, n):
        # plot_predictions(y_pred.iloc[:num_rows,i], y_test.iloc[:num_rows,i], y_ct.iloc[:num_rows,i], datetime_string, "combined_cyc_10_20_std_" + headers[i].split("_")[0])
        plot_predictions(y_pred.iloc[:num_rows,i], y_test.iloc[:num_rows,i], datetime_string[:num_rows], "Cyclone 3 " + headers[i].split("_")[0])

    
    scaler = StandardScaler()

    # Normalize y_pred and y_test
    y_test_norm = scaler.fit_transform(y_test.iloc[:num_rows,1:])
    y_pred_norm = scaler.transform(y_pred.iloc[:num_rows,1:])

    # Compute the mean squared error (MSE) between the normalized y_pred and y_test
    mse = mean_squared_error(y_test_norm, y_pred_norm)

    print('The Mean Squared Error (MSE) is:', mse)
