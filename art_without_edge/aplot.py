from collections import OrderedDict

import numpy as np
from matplotlib import pyplot as plt


def plot_line_interpretability_error_rate_tradeoff_from_coords(err_train, err_test, x_label=None,
                                                               y_label=None, file_path=None, title=None,
                                                               xlim=None, grid=True):
    """Plot an interpretability error rate tradeoff from coordinates

    Args:
        err_train (list): List of tuples (x_value, err_train_value_at_x)
        err_test (list): List of tuples (x_value, err_test_value_at_x)
        x_label (str): Name of the X-axis label
        y_label (str): Name of the Y-axis label
        file_path (str): Path of the file where the plot will be saved
        title (str): Title of the plot
        xlim (tuple): X-axis domain shown
        grid (bool): If true then show a grid
    """
    err_train = list(set(err_train))
    err_train.sort()
    for i in range(len(err_train)):
        err_train[i] = list(err_train[i])  # tuple to list
    err_train = np.array(err_train)

    err_test = list(set(err_test))
    err_test.sort()
    for i in range(len(err_test)):
        err_test[i] = list(err_test[i])  # tuple to list
    err_test = np.array(err_test)

    if len(err_train) != 0:
        plt.plot(err_train[:, 0], err_train[:, 1], c='darkorange', marker='o', label="Train")
    if len(err_test) != 0:
        plt.plot(err_test[:, 0], err_test[:, 1], c='blue', marker='o', label="Test")
    #plt.xlabel(x_label)
    if title is not None:
        plt.title(title)
    if grid:
        plt.grid()

    #plt.ylabel(y_label)
    plt.ylim(0, 1)
    plt.xlim(0,20)
    if xlim is not None:
        plt.xlim(xlim)

    if len(err_train) != 0 or len(err_test) != 0:
        plt.legend(loc="upper left")

    if file_path is not None:
        plt.savefig(file_path)

    plt.show()


import csv

# CSVファイルのパス
csv_file = "C:/Users/Ayato Tomofuji/Documents/Mof/MoFGBMLPy/results/2/vehicle_cnn/a0_3_cnn/results.csv"

# リストの初期化
training_error_rates = []
test_error_rates = []
num_rules_list = []

# CSVファイルを読み込む
with open(csv_file, newline='', encoding='utf-8') as file:
    reader = csv.DictReader(file)

    for row in reader:
        training_error_rates.append((int(row["num_rules"]), float(row["training_error_rate"])))
        test_error_rates.append((int(row["num_rules"]), float(row["test_error_rate"])))
# 出力（確認用）
print(training_error_rates)
print(test_error_rates)
print(num_rules_list)
plot_line_interpretability_error_rate_tradeoff_from_coords(training_error_rates, test_error_rates, x_label=None,
                                                               y_label=None, file_path=None, title=None,
                                                               xlim=None, grid=True)