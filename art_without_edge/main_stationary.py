import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine, make_blobs
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
data = scipy.io.loadmat('2D_ClusteringDATASET.mat')['data']
numD = 5000


# Segment data
d1 = data[:numD]
d2 = data[15000:15000 + numD]
d3 = data[30000:30000 + numD]
d4 = data[45000:45000 + numD]
d5 = data[60000:60000 + numD]
d6 = data[75000:75000 + numD]
data = np.vstack([d1, d2, d3, d4, d5, d6])



# Normalization [0-1]
# Normalization [0-1]
scaler = MinMaxScaler()
data = scaler.fit_transform(data)
# Randomize data
np.random.seed(11)
data = shuffle(data)

# Noise Setting [0,1]
if NR > 0:
    noise_data = np.random.rand(int(data.shape[0] * NR), data.shape[1])
    data[:len(noise_data)] = noise_data

# Parameters ========================================================
net = ARTNet(Lambda=50, minCIM=0.15)
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

estimateDensityByCountNode(net)



