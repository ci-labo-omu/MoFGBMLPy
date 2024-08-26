import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from art_without_edge.main_stationary import estimateDensityByCountNode, ARTclustering_woEdge_Train
from art_without_edge.myPlot_withoutEdge import myPlot_withoutEdge

# Initialize network parameters
net = {
    'numNodes': 0,      # Number of nodes
    'weight': [],       # Node position
    'CountNode': [],    # Winner counter for each node
    'adaptiveSig': [],  # Kernel bandwidth for CIM in each node
    'edge': np.zeros((2, 2)),  # Initial connections (edges) matrix
    'LabelCluster': [],
    'Lambda': 50,       # Interval for calculating a kernel bandwidth for CIM
    'minCIM': 0.3       # Similarity threshold
}

# Constants
TRIAL = 1    # Number of trials
NR = 0.0     # Noise rate [0-1]
c = 6        # Number of clusters
numD = 5000  # Number of data points

# Load data
# Assumed the dataset is loaded into `data`
# For example, using NumPy to load:
# data = np.loadtxt('2D_ClusteringDATASET.txt')
data = np.random.rand(90000, 2)  # Placeholder, replace with actual data

DATA = data[:, :2]
originDATA = DATA.copy()

# Prepare data slices
d1 = DATA[:numD, :]
d2 = DATA[15001:15000+numD, :]
d3 = DATA[30001:30000+numD, :]
d4 = DATA[45001:45000+numD, :]
d5 = DATA[60001:60000+numD, :]
d6 = DATA[75001:75000+numD, :]
DATA = np.vstack((d1, d2, d3, d4, d5, d6))

# Normalization [0-1]
scaler = MinMaxScaler()
DATA = scaler.fit_transform(DATA)

time_train = 0

for trial in range(1, TRIAL * c + 1):
    print(f'Iterations: {trial}/{TRIAL * c}')

    idx = (trial - 1) % c + 1
    data = DATA[(idx-1)*numD:idx*numD, :]

    # Noise setting [0, 1]
    if NR > 0:
        noiseDATA = np.random.rand(int(data.shape[0] * NR), data.shape[1])
        data[:noiseDATA.shape[0], :] = noiseDATA

    # Randomize data
    np.random.seed(trial)
    np.random.shuffle(data)

    # Training ==========================================
    import time
    start_time = time.time()
    net = ARTclustering_woEdge_Train(data, net)  # Placeholder for the training function
    time_train += time.time() - start_time
    # ===================================================

    # Results
    print(f'   Num. Clusters: {net["numNodes"]}')
    print(f' Processing Time: {time_train}\n')

    # Plotting ==========================================
    plt.figure(figsize=(12, 6))
    plt.clf()

    # Plot network
    plt.subplot(1, 2, 1)
    myPlot_withoutEdge(data, net)  # Placeholder for the plotting function

    # Plot estimated density
    plt.subplot(1, 2, 2)
    estimateDensityByCountNode(net)  # Placeholder for density estimation function
    plt.axis('square')

    plt.show()
    plt.draw()
    # ===================================================
