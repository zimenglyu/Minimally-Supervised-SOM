# plot code to compare predicted vs actual values
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_predictions(y_pred, y_test, title):
    plt.figure(figsize=(20, 10 ))
    plt.plot(y_test, label='Coal Tracker')
    plt.plot(y_pred, label='Predicted')
    plt.legend()
    plt.title(title)
    plt.savefig(title + '.png')


# main function
if __name__ == '__main__':
    #  load data
    # pred_path = "/Users/zimenglyu/Documents/code/git/susi/2022_09_10_lab_results_prediction.csv"
    # test_path = "/Users/zimenglyu/Documents/datasets/microbeam/PPM/2022-09/MGA_10_2022_09_lab_results.csv"
    # pred_path = "/Users/zimenglyu/Documents/code/git/susi/2022_09_prediction.csv"
    # test_path = "/Users/zimenglyu/Documents/datasets/microbeam/PPM/2022-09/MGA_10_2022_09_clean.csv"
    pred_path = "/Users/zimenglyu/Documents/code/git/susi/Sep_2022_cyc_10_to_10_prediction.csv"
    test_path = "/Users/zimenglyu/Documents/datasets/microbeam/PPM/2022-09/MGA_10_2022_09_clean_roll_10.csv"
    y_pred = pd.read_csv(pred_path)
    y_test = pd.read_csv(test_path)
    headers = y_pred.columns
    print(headers)
    # plot predictions
    n = y_pred.shape[1]
    
    for i in range(n):
        plot_predictions(y_pred.iloc[:,i], y_test.iloc[:,i], "Sep_2022_cyc_10_10_" + headers[i])
