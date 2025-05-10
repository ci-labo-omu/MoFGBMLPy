import pandas as pd
import matplotlib.pyplot as plt
import os
from collections import defaultdict
import numpy as np



data_name = "cancer"
# データが保存されているディレクトリ
# fold数
num_folds = 30
def withdensity(data_name):
    print(data_name)
    data_dir = f"C:/Users/Ayato Tomofuji/Documents/Mof/MoFGBMLPy/art_without_edge/dataset_nodes/{data_name}"  # ここを実際のディレクトリに変更

    # 各foldファイルを読み込み（ファイル名は fold_0.csv ~ fold_29.csv を想定）
    fold_names = [f"a{i}_{j}" for i in range(3) for j in range(10)]
    node_set = (20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75)
    print(node_set)
    node_stats = defaultdict(lambda: {'sum': 0.0, 'sum_sq': 0.0, 'count': 0})

    for node in sorted(node_set):
        for i in range(3):         # a0, a1, a2
            for j in range(10):    # -0 ～ -9
                fold_id = f"a{i}_{j}"

                file_name = f"a{i}_{j}_{data_name}_tra/{fold_id}_{data_name}_node{node}.csv"
                file_path = os.path.join(data_dir, file_name)

                if not os.path.exists(file_path):
                    print(f"ファイルが見つかりません: {file_path}")
                    continue

                with open(file_path, 'r') as f:
                    header = f.readline().strip().split(',')
                    num_patterns = int(header[0])
                node_stats[node]['sum'] += num_patterns
                node_stats[node]['sum_sq'] += num_patterns**2
                node_stats[node]['count'] += 1

    node_exec_summary = {}

    for node, stats in node_stats.items():
        n = stats['count']
        if n > 0:
            mean = stats['sum'] / n
            variance = (stats['sum_sq'] / n) - (mean ** 2)
            std_dev = np.sqrt(variance) if variance > 0 else 0.0
            node_exec_summary[node] = {'mean': mean, 'std': std_dev}
        else:
            node_exec_summary[node] = {'mean': None, 'std': None}
    # 結果表示
    for node, summary in node_exec_summary.items():
        print(f"Node {node}: mean = {summary['mean']:.4f}, std = {summary['std']:.4f}")

def withcnn(data_name):
    data_dir = f"C:/Users/Ayato Tomofuji/Documents/Mof/MoFGBMLPy/art_without_edge/nodes_cnn/{data_name}"

    fold_names = [f"a{i}_{j}" for i in range(3) for j in range(10)]

    pattern_counts = []

    for fold_id in fold_names:
        file_name = f"{fold_id}_{data_name}_cnn.dat"
        file_path = os.path.join(data_dir, file_name)

        if not os.path.exists(file_path):
            print(f"ファイルが見つかりません: {file_path}")
            continue

        with open(file_path, 'r') as f:
            first_line = f.readline().strip()
            try:
                num_patterns = int(first_line.split(',')[0])
                pattern_counts.append(num_patterns)
            except ValueError:
                print(f"数値の読み込みに失敗: {file_path}")

    if pattern_counts:
        mean = np.mean(pattern_counts)
        std = np.std(pattern_counts, ddof=0)  # 母集団標準偏差（通常はこちらでOK）
        print(f"平均: {mean:.4f}, 標準偏差: {std:.4f}")
    else:
        print("有効なパターン数が取得できませんでした。")
withdensity(data_name)
