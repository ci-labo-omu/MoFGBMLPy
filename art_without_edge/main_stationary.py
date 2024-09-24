import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine, make_blobs, fetch_openml, make_classification
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
import scipy.io

from art_without_edge.ARTclustering_woEdge_Train import ARTNet
from art_without_edge.estimateDensityByCoountNode import estimateDensityByCountNode
from art_without_edge.myPlot_withoutEdge import myPlot_withoutEdge

# Number of trials
TRIAL = 1

# Noise Rate [0-1]
NR = 0.0

# Load 2D_ClusteringDATASET
data = scipy.io.loadmat('2D_ClusteringDATASET.mat')['data']  # OpenMLデータセットの読み込み
data, y = make_classification(n_samples=8000,  # サンプル数
                              n_features=3 ,  # 特徴量の数（2つの特徴量）
                              flip_y=0,
                              class_sep=2.2,
                              n_informative=2,  # 有益な特徴量の数
                              n_redundant=0,  # 冗長な特徴量の数
                              n_clusters_per_class=1,  # クラスごとのクラスター数
                              n_classes=4,  # クラス数（4クラス分類）
                              random_state=42)  #
#それぞれのクラスのデータ数を表示
print(np.unique(y, return_counts=True))
data = MinMaxScaler().fit_transform(data)
# dataを2次元平面でプロット再現性のための乱数シード
# dataとyを結合
data = np.hstack([data, y.reshape(-1, 1)])

"""
numD = 5000


# Segment data
d1 = data[:numD]
d2 = data[15000:15000 + numD]
d3 = data[30000:30000 + numD]
d4 = data[45000:45000 + numD]
d5 = data[60000:60000 + numD]
d6 = data[75000:75000 + numD]
data = np.vstack([d1, d2, d3, d4, d5, d6])

y1 = y[:numD]
y2 = y[15000:15000 + numD]
y3 = y[30000:30000 + numD]
y4 = y[45000:45000 + numD]
y5 = y[60000:60000 + numD]
y6 = y[75000:75000 + numD]
y = np.hstack([y1, y2, y3, y4, y5, y6])"""



data_1 = data[y == 0]
data_2 = data[y == 1]
data_3 = data[y == 2]
data_4 = data[y == 3]



for i, data in enumerate([data_1, data_2, data_3, data_4]):
    # Normalization [0-1]
    # Normalization [0-1]

    # Randomize data
    np.random.seed(11)
    data = shuffle(data)
    # Noise Setting [0,1]
    #     if NR > 0:
    #         noise_data = np.random.rand(int(data.shape[0] * NR), data.shape[1])
    #         data[:len(noise_data)] = noise_data
    #
    #     # Parameters ========================================================
    net = ARTNet(Lambda=50, minCIM=0.10)
    # ====================================================================

    time_train = 0

    for trial in range(TRIAL):
        print(f'Iterations: {trial + 1}/{TRIAL}')

        # Randomize data
        data = shuffle(data)

        # Training ==========================================
        start_time = time.time()
        net.ARTclustering_woEdge_Train(data)
        time_train += time.time() - start_time
        # ===================================================

        # Results
        resultNumNodes = f'   Num. Clusters: {net.numNodes}'
        print(resultNumNodes)
        print(f' Processing Time: {time_train}')
        print('')

    myPlot_withoutEdge(data, net)




    #ノード座標と数をカウントする
    estimateDensityByCountNode(net, i)
