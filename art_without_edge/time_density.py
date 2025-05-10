import pandas as pd
import matplotlib.pyplot as plt
import os
from collections import defaultdict
import numpy as np

from art_without_edge.num_pattern import withdensity

data_name = "cancer"
# データが保存されているディレクトリ
data_dir = f"C:/Users/Ayato Tomofuji/Documents/Mof/MoFGBMLPy/results/1/{data_name}_density"  # ここを実際のディレクトリに変更

# fold数
num_folds = 30

# 各ルール数ごとのfold出現回数とエラーレート記録用
rule_fold_count = defaultdict(set)
rule_training_errors = defaultdict(list)
rule_test_errors = defaultdict(list)
fold_names = [f"a{i}_{j}" for i in range(3) for j in range(10)]
print(data_dir)
node_set = set()
for fname in os.listdir(data_dir):
    for fold in fold_names:
        if fname.startswith(fold + f"_{data_name}_node"):
            try:
                node = int(fname.split("_node")[-1])
                node_set.add(node)
            except:
                pass
# node数ごとの処理
node_stats = defaultdict(lambda: {'sum': 0.0, 'sum_sq': 0.0, 'count': 0})

node_exec_avg = {}
for node in sorted(node_set):
    rule_count = defaultdict(int)
    for fold in fold_names:
        folder = f"{fold}_{data_name}_node{node}"
        file_path = os.path.join(data_dir, folder, 'exec_time.txt')
        with open(file_path, 'r') as f:
            try:
                value = float(f.read().strip())
                node_stats[node]['sum'] += value
                node_stats[node]['sum_sq'] += value**2
                node_stats[node]['count'] += 1
            except ValueError:
                print(f"Invalid float in {file_path}")

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
print(data_name)
for node, summary in node_exec_summary.items():
    print(f"Node {node}: mean = {summary['mean']:.4f}s, std = {summary['std']:.4f}s")

