# plot code to compare predicted vs actual values
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def plot_predictions(y_pred, datetime, title):
    fig, ax = plt.subplots(1,1, figsize=(20, 10 ))
    x = np.arange(len(y_pred))
    ax.plot(x, y_pred, label='Predicted')
    # ax.plot(x, y_test, label='Coal Tracker')
    # ax.plot(x, y_ct, label='CT')
    # plt.gcf().autofmt_xdate()
    ax.set_xticks(x)
    ax.set_xticklabels([datetime[i] for i in x])
    ax.set_xticklabels(datetime, rotation=90, ha='right')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(fontsize = 17)
    # plt.xlabel('Timestamp')
    plt.title(title, fontsize = 20)
    plt.tight_layout()
    plt.savefig(title + '.png')

# main function
if __name__ == '__main__':

    pred_path = "/Users/zimenglyu/Documents/code/git/susi/2023-04_20_5_standard_per_minute_prediction.csv"

    y_pred = pd.read_csv(pred_path)

    y_pred['DateTime']= pd.to_datetime(y_pred['DateTime'])
    datetime_string = y_pred['DateTime']
    # print(datetime_string)
    # y_test.set_index('DateTime', inplace=True)
    headers = y_pred.columns
    # print(headers)
    # plot predictions
    n = y_pred.shape[1]
    # num_rows = min(y_pred.shape[0], y_test.shape[0])
 
    for i in range(1, n):
        # plot_predictions(y_pred.iloc[:num_rows,i], y_test.iloc[:num_rows,i], y_ct.iloc[:num_rows,i], datetime_string, "202303_202105_202209_cyc_10_20_std_" + headers[i].split("_")[0])
        plot_predictions(y_pred.iloc[:,i], datetime_string[:], "2023-04_per_minute_" + headers[i].split("_")[0])

