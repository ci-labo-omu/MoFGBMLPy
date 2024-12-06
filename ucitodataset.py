import re

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import MinMaxScaler

from mofgbmlpy.data.input import Input
from sklearn.model_selection import KFold
from mofgbmlpy.main.nsgaii.mofgbml_nsgaii_main import MoFGBMLNSGAIIMain
from ucimlrepo import fetch_ucirepo

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import numpy as np
import pandas as pd

# データ取得
from ucimlrepo import fetch_ucirepo
dataset = fetch_ucirepo(id=264)
X = dataset.data.features.to_numpy()
y = dataset.data.targets.to_numpy()

# 必要に応じてラベルをエンコード
le = LabelEncoder()
y = le.fit_transform(y)

# 正規化
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# データを結合
data = np.hstack((X, y.reshape(-1, 1)))

# データ保存
num_pattern, num_feature = X.shape
num_class = len(np.unique(y))
output_path = "dataset/eeg/eeg.dat"

with open(output_path, "w") as f:
    f.write(f"{num_pattern},{num_feature},{num_class}\n")
    np.savetxt(f, data, delimiter=",", fmt="%.6f")

print(f"Dataset saved to {output_path}")