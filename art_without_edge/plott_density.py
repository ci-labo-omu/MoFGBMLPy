import pandas as pd
import matplotlib.pyplot as plt
import os
from collections import defaultdict
import numpy as np



data_name = "vehicle"
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
summary_by_node = {}


for node in sorted(node_set):
    print(node)
    rule_stats = defaultdict(lambda: {'train': [], 'test': []})
    rule_count = defaultdict(int)
    valid_fold_count = defaultdict(int)
    for fold in fold_names:
        folder = f"{fold}_{data_name}_node{node}"
        csv_path = os.path.join(data_dir, folder, 'results.csv')
        if os.path.exists(csv_path):
            print(csv_path)
            df = pd.read_csv(csv_path)
            seen_rules = set()
            for _, row in df.iterrows():
                r = row['num_rules']
                rule_stats[r]['train'].append(row['training_error_rate'])
                rule_stats[r]['test'].append(row['test_error_rate'])
                seen_rules.add(r)
            for r in seen_rules:
                valid_fold_count[r] += 1

    # 16回以上出現したルールだけ残す
    rule_avg = {
        r: {
            'train': sum(rule_stats[r]['train']) / len(rule_stats[r]['train']),
            'test': sum(rule_stats[r]['test']) / len(rule_stats[r]['test']),
        }
        for r in rule_stats
        if valid_fold_count[r] >= 16
    }

    summary_by_node[node] = rule_avg
# グラフ描画
print(summary_by_node.items())
for node, results in sorted(summary_by_node.items()):

    rules = sorted(results.keys())
    plt.ylim(0, 1)
    plt.xlim(0,12)
    plt.grid(True)
    plt.tight_layout()
    #plt.title(f"MinCIM = {int(node)/100}")
    test_errors = [results[r]['test'] for r in rules]
    train_errors = [results[r]['train'] for r in rules]
    plt.plot(rules, train_errors, marker='o', c="darkorange", label="Train")
    plt.plot(rules, test_errors, marker='o', c="blue", label="Test")


    plt.legend()
    plt.show()



