import pandas as pd
import matplotlib.pyplot as plt
import os
from collections import defaultdict
import numpy as np



data_name = "vehicle"
# データが保存されているディレクトリ
data_dir = f"C:/Users/Ayato Tomofuji/Documents/Mof/MoFGBMLPy/results/1/{data_name}"  # ここを実際のディレクトリに変更

# fold数
num_folds = 30

# 各ルール数ごとのfold出現回数とエラーレート記録用
rule_fold_count = defaultdict(set)
rule_training_errors = defaultdict(list)
rule_test_errors = defaultdict(list)

sum_time = 0.0

# 各foldファイルを読み込み（ファイル名は fold_0.csv ~ fold_29.csv を想定）
for i in range(3):         # a0, a1, a2
    for j in range(10):    # -0 ～ -9
        fold_id = f"{i}_{j}"

        file_name = f"a{i}_{j}/exec_time.txt"
        file_path = os.path.join(data_dir, file_name)

        if not os.path.exists(file_path):
            print(f"ファイルが見つかりません: {file_path}")
            continue

        with open(file_path, 'r') as f:
            value = float(f.read())
        sum_time += value


ave_time = sum_time / 30

