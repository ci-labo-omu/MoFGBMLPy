import numpy as np
from imblearn.under_sampling import CondensedNearestNeighbour
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from pathlib import Path

# 実験設定
data_name = "cancer"
train_dir = Path(f"../dataset/{data_name}/")
output_base = Path(f"./nodes_cnn/{data_name}/")
output_base.mkdir(parents=True, exist_ok=True)

for train_file in train_dir.glob(f"*{data_name}-10tra.dat"):
    print(f"Processing {train_file.name}")
    identifier = train_file.stem.split(f"-10tra")[0]

    with open(train_file, 'r') as f:
        header = f.readline().strip().split(',')
        num_rows = int(header[0])
        num_dims = int(header[1])
        num_classes = int(header[2])

        X = np.zeros((num_rows, num_dims))
        y = np.empty(num_rows, dtype=object)
        for i, line in enumerate(f):
            data = line.rstrip(',\n').split(',')
            X[i] = np.array(data[:-1], dtype=float)
            y[i] = data[-1]

    # ラベルを数値化（CNNは数値ラベルしか扱えない）
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # CNNで圧縮
    print("  -> applying Condensed Nearest Neighbour...")
    cnn = CondensedNearestNeighbour(n_neighbors=1, random_state=42)
    X_resampled, y_resampled = cnn.fit_resample(X, y_encoded)

    # ラベルを元に戻す
    y_resampled_str = le.inverse_transform(y_resampled)

    # 保存ファイルパス
    output_file = output_base / f"{identifier}_cnn.dat"

    with open(output_file, 'w') as out_f:
        out_f.write(f"{len(X_resampled)},{num_dims},{num_classes}\n")
        for x_row, y_val in zip(X_resampled, y_resampled_str):
            x_str = ','.join(map(str, x_row))
            out_f.write(f"{x_str},{y_val}\n")

    print(f"  -> saved to {output_file}")