import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine, make_blobs, fetch_openml, make_classification
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
import scipy.io

from art_without_edge.ARTclustering_woEdge_Train import ARTNet
from art_without_edge.estimateDensityByCountNode import estimateDensityByCountNode
from art_without_edge.myPlot_withoutEdge import myPlot_withoutEdge, myPlot_color

# Number of trials
TRIAL = 1

# Noise Rate [0-1]
NR = 0.0

# Load 2D_ClusteringDATASET
#data = scipy.io.loadmat('2D_ClusteringDATASET.mat')['data']  # OpenMLデータセットの読み込み

data_name = "banknote"
train_dir = Path(f"../dataset/{data_name}/")

for train_file in train_dir.glob(f"*{data_name}-10tra.dat"):
    print(train_file)
    identifier = train_file.stem.split(f"-10tra")[0]
    with open(train_file, 'r') as f:
        print(train_file)
        #ヘッダ行はサンプル数，次元数，クラス数の3つの整数をカンマ区切りで記述されている
        header = f.readline().strip().split(',')
        num_rows = int(header[0])
        num_dims = int(header[1])
        num_classes = int(header[2])
        print(num_rows, num_dims, num_classes)
        X = np.zeros((num_rows, num_dims))
        y = np.zeros(num_rows)
        for i, line in enumerate(f):
            data = line.strip().split(',')
            X[i] = np.array(data[:-1], dtype=float)
            y[i] = data[-1]

        #それぞれのクラスのデータ数を表示
        #dataとyをdatファイルに書き出し，それぞれの行を横に並べて
        data = np.hstack([X, y.reshape(-1, 1)])
        # dataを2次元平面でプロット再現性のための乱数シード
        # dataとyを結合x
        #dataをプロット
        """
        plt.scatter(data[y == 0, 0], data[y == 0, 1], label='Class 1', s=50, alpha=0.6)
        plt.scatter(data[y == 1, 0], data[y == 1, 1], label='Class 2', s=50, alpha=0.6)
        plt.scatter(data[y == 2, 0], data[y == 2, 1], label='Class 3', s=50, alpha=0.6)
        plt.scatter(data[y == 3, 0], data[y == 3, 1], label='Class 4', s=50, alpha=0.6)
        plt.legend()
        #plt.show()
        
        numD = 5000
        
        
        # vehicle data
        d1 = data[:numD]
        d2 = data[15000:15000 + numD]
        d3 = data[30000:3000]. b0 + numD]
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

        data_by_class = {label: data[y == label] for label in np.unique(y)}


        minCIMs = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
        for minCIM in minCIMs:
            for i, data in data_by_class.items():
                # Normalization [0-1]
                # Normalization [0-1]
                # Randomize data
                np.random.seed(11)
                data = shuffle(data)

                # Noise Setting [0,1]
                #     if NR > 0:
                #      %   noise_data = np.random.rand(int(data.shape[0] * NR), data.shape[1])
                #         data[:len(noise_data)] = noise_data
                #
                #     # Parameters ========================================================
                net = ARTNet(Lambda=30, minCIM=minCIM)
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
                #myPlot_withoutEdge(data, net)
                #ノード座標と数をカウントする
                estimateDensityByCountNode(net, minCIM, data_name, identifier)
