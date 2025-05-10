import os
import pandas as pd

# ベースディレクトリを指定
base_directory = "dataset_nodes/blood/"

# ベースディレクトリ以下を再帰的に探索
for root, _, files in os.walk(base_directory):
    for filename in files:
        if filename.endswith(".csv"):  # CSVファイルだけを処理
            filepath = os.path.join(root, filename)

            # CSVファイルを読み込む（最初のヘッダー行をスキップ）
            df = pd.read_csv(filepath, header=None, skiprows=1)

            # 行数、次元数（列数 - 2）、クラス数を計算
            num_rows = len(df)  # 行数
            num_dims = df.shape[1] - 2  # 列数 - 2
            num_classes = df.iloc[:, -2].nunique()  # 最後から2列目でユニークなクラス数を計算

            # ヘッダ行を作成
            header = f"{num_rows},{num_dims},{num_classes}"

            # データフレームを保存（新しいファイルにヘッダを挿入）
            with open(filepath, "w", newline='') as f:
                f.write(header + "\n")  # ヘッダ行を追加
                df.to_csv(f, header=False, index=False)  # データを書き込む

print("ヘッダの追加が完了しました。")