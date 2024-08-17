import copy
import csv
import os
import time
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from matplotlib import pyplot as plt

from mofgbmlpy.fuzzy.knowledge.factory.homo_triangle_knowledge_factory_2_3_4_5 import \
    HomoTriangleKnowledgeFactory_2_3_4_5
from mofgbmlpy.main.abstract_mofgbml_main import AbstractMoFGBMLMain
from mofgbmlpy.main.nsgaii.mofgbml_nsgaii_main import MoFGBMLNSGAIIMain


def process_runs_results(runs_results, x_key="total_rule_length", y_key="training_error_rate",
                         keep_empty_x_key_values=False,
                         remove_rare_solutions=True):
    """Process the runs results

    Args:
        runs_results (dict): Results as a dictionary
        x_key (str): Key in the dict for the X-axis (e.g. num_rules)
        y_key (str): Key in the dict for the Y-axis (e.g. training_error_rate)
        keep_empty_x_key_values (bool): If true then keep the keys with no values in the results dict
        remove_rare_solutions (): If true then remove solutions with an interpretability value that appears in less than 50 % of the results

    Returns:
        dict: The processed results
    """

    if x_key != "total_rule_length" and x_key != "num_rules":
        raise Exception("only total_rule_length and num_rules are accepted for the x_key")

    data = {}
    x_occ = {}
    max_x = float('-inf')
    num_runs = 0

    for run in runs_results:
        num_runs += 1
        for res in run:
            x = int(res[x_key])

            if x not in data:
                data[x] = []
                x_occ[x] = 0

            x_occ[x] += 1
            data[x].append(float(res[y_key]))
            if x > max_x:
                max_x = x

    new_data = {}
    for i in range(int(max_x)):
        if i not in data:
            if keep_empty_x_key_values:
                new_data[i] = []
        elif remove_rare_solutions and x_occ[i] < num_runs // 2:
            if keep_empty_x_key_values:
                new_data[i] = []
        else:
            new_data[i] = data[i]

    return OrderedDict(sorted(new_data.items()))


def show_results_median_line_plot(runs_results, x_key, remove_rare_solutions=True, xlim=None):
    """Show the results in a median line plot after aggregating them

    Args:
        runs_results (dict): Results as a dictionary
        x_key (str): Key in the dict for the X-axis (e.g. num_rules)
        remove_rare_solutions (): If true then remove solutions with an interpretability value that appears in less than 50 % of the results
        xlim (tuple): A Pair of numbers specifying the x-axis limits
    """
    data_train = process_runs_results(runs_results, x_key=x_key, y_key="training_error_rate",
                                      remove_rare_solutions=remove_rare_solutions)
    data_test = process_runs_results(runs_results, x_key=x_key, y_key="test_error_rate",
                                     remove_rare_solutions=remove_rare_solutions)
    err_train = []
    err_test = []

    for x, y_vals in data_train.items():
        err_train.append((x, np.median(y_vals)))
    for x, y_vals in data_test.items():
        err_test.append((x, np.median(y_vals)))

    AbstractMoFGBMLMain.plot_line_interpretability_error_rate_tradeoff_from_coords(err_train,
                                                                                   err_test,
                                                                                   x_label=x_key,
                                                                                   y_label="error_rate",
                                                                                   xlim=xlim)


def show_results_box_plot(runs_results, x_key, remove_rare_solutions=True, title=None):
    """Show the results using box plots after aggregating them

    Args:
        runs_results (dict): Results as a dictionary
        x_key (str): Key in the dict for the X-axis (e.g. num_rules)
        remove_rare_solutions (): If true then remove solutions with an interpretability value that appears in less than 50 % of the results
        title (str): Title for the plot
    """
    data = process_runs_results(runs_results, x_key=x_key, y_key="training_error_rate", keep_empty_x_key_values=True,
                                remove_rare_solutions=remove_rare_solutions)

    fig, ax = plt.subplots()
    labels = list(data.keys())
    ax.boxplot(data.values(), labels=labels)
    plt.xlabel(x_key)
    plt.ylabel("training_error_rate")
    plt.ylim([0, 1])
    if title is not None:
        plt.title(title)

    plt.show()


def task(args):
    """Task for the parallel cross validation test: runs MoFGBMLPy on one Arguments object.
    Note that for now only NSGA-II is used

    Args:
        args (Arguments): Arguments object used by the runner
    """
    runner = MoFGBMLNSGAIIMain(HomoTriangleKnowledgeFactory_2_3_4_5)
    runner.main(args)


def load_result_csv(path):
    """Load a CSV file from path as a dictionary

    Args:
        path (str): CSV file path

    Returns:
        dict: CSV data
    """
    result = []
    with open(path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            result.append(row)
    return result


def load_results_csv(paths):
    """Load all the CSV files in the paths given and add the loaded data to a list

    Args:
        paths (str[]): List of CSV file paths

    Returns:
        list: List of CSV data (one item is one file data)
    """
    results = []
    for path in paths:
        results.append(load_result_csv(path))
    return results


def run_cross_validation(args, dataset_root):
    """Run a cross validation test on a dataset using pre-split dataset files and save the results in files

    Args:
        args (Arguments): Arguments object
        dataset_root (str): Path to the dataset root directory:
    """
    start = time.time()

    data_name_arg_idx = args.index("--data-name") + 1
    data_name = args[data_name_arg_idx]

    runs_args = [
        args + ["--train-file", f"{dataset_root}/{data_name}/a{i}_{j}_{data_name}-10tra.dat",
                "--test-file", f"{dataset_root}/{data_name}/a{i}_{j}_{data_name}-10tst.dat",
                "--experiment-id", f"trial{i}{j}",
                ]
        for i in range(3) for j in range(10)]

    task(runs_args[0])

    with ProcessPoolExecutor() as executor:
        executor.map(task, runs_args)

    print("Execution time:", time.time() - start)


def get_results(root_folder, algorithm_id, data_name):
    """Get all the results from a root results folder (CSV files)

    Args:
        root_folder (str): Root results folder
        algorithm_id (str): Algorithm ID
        data_name (str): Name of the dataset, e.g. Iris

    Returns:
        list: List of results
    """
    results_path = root_folder + os.sep + algorithm_id + os.sep + data_name
    runs_results_folders = [f"{results_path}/trial{i}{j}/results.csv" for i in range(3)
                            for j in range(10)]

    return load_results_csv(runs_results_folders)
