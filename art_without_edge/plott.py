import pandas as pd
import matplotlib.pyplot as plt
import os
from collections import defaultdict
import numpy as np



data_name = "cancer"
# データが保存されているディレクトリ
data_dir = f"C:/Users/Ayato Tomofuji/Documents/Mof/MoFGBMLPy/results/1/{data_name}"  # ここを実際のディレクトリに変更

# fold数
num_folds = 30

# 各ルール数ごとのfold出現回数とエラーレート記録用
rule_fold_count = defaultdict(set)
rule_training_errors = defaultdict(list)
rule_test_errors = defaultdict(list)

# 各foldファイルを読み込み（ファイル名は fold_0.csv ~ fold_29.csv を想定）
for i in range(3):         # a0, a1, a2
    for j in range(10):    # -0 ～ -9
        fold_id = f"{i}_{j}"

        file_name = f"a{i}_{j}/results.csv"
        file_path = os.path.join(data_dir, file_name)

        if not os.path.exists(file_path):
            print(f"ファイルが見つかりません: {file_path}")
            continue

        df = pd.read_csv(file_path)

        # fold内で登場したルール数
        seen_rules = set()

        for _, row in df.iterrows():
            rules = int(row["num_rules"])
            train_err = row["training_error_rate"]
            test_err = row["test_error_rate"]

            rule_training_errors[rules].append(train_err)
            rule_test_errors[rules].append(test_err)
            seen_rules.add(rules)

        for rule in seen_rules:
            rule_fold_count[rule].add(fold_id)

# 有効なルール数を選別（30fold中16fold以上に出現）
valid_rules = [rule for rule, folds in rule_fold_count.items() if len(folds) >= 16]

# 平均エラーレートを計算
mean_train_errors = {rule: np.mean(rule_training_errors[rule]) for rule in valid_rules}
mean_test_errors = {rule: np.mean(rule_test_errors[rule]) for rule in valid_rules}

# ルール数でソート
sorted_rules = sorted(valid_rules)

# グラフ描画
plt.figure()
plt.plot(sorted_rules, [mean_train_errors[r] for r in sorted_rules], label="Train", marker="o", c='darkorange')
plt.plot(sorted_rules, [mean_test_errors[r] for r in sorted_rules], label="Test", marker="o", c='blue')
#plt.xlabel("Number of Rules")
#plt.ylabel("Error Rate")
plt.legend()
plt.grid(True)
plt.ylim(0,1)
plt.xlim(0,12)
plt.tight_layout()
plt.show()