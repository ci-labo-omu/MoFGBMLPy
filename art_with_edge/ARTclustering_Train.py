import numpy as np
import networkx as nx  # For graph-based clustering (conncomp equivalent)

class ARTNet:
    def __init__(self, numNodes=0, weight=None, LabelCluster=None, CountNode=None, adaptiveSig=None, Lambda=50, minCIM=0.15):
        if weight is None:
            weight = []
        if CountNode is None:
            CountNode = []
        if adaptiveSig is None:
            adaptiveSig = []
        if LabelCluster is None:
            LabelCluster = []

        self.edge = np.zeros((2, 2))  # This will be resized as needed
        self.numNodes = numNodes
        self.weight = weight
        self.CountNode = CountNode
        self.adaptiveSig = adaptiveSig
        self.Lambda = Lambda
        self.minCIM = minCIM
        self.LabelCluster = LabelCluster

    def ARTclustering_Train(self, DATA):
        numNodes = self.numNodes
        weight = self.weight
        CountNode = self.CountNode
        adaptiveSig = self.adaptiveSig
        Lambda = self.Lambda
        minCIM = self.minCIM
        edge = self.edge

        for sampleNum in range(DATA.shape[0]):
            if len(weight) == 0 or sampleNum % Lambda == 0:
                estSigCA = self.SigmaEstimation(DATA, sampleNum, Lambda)

            input_data = DATA[sampleNum, :]

            if len(weight) < 1:
                # Add Node
                numNodes += 1
                weight.append(input_data)
                CountNode.append(1)
                adaptiveSig.append(estSigCA)
                edge = self._add_new_node(edge, numNodes)
            else:
                globalCIM = self.CIM(input_data, np.array(weight), np.mean(adaptiveSig))
                Lcim_s1, s1 = np.min(globalCIM), np.argmin(globalCIM)
                globalCIM[s1] = np.inf
                Lcim_s2, s2 = np.min(globalCIM), np.argmin(globalCIM)

                if minCIM < Lcim_s1:
                    # Add Node
                    numNodes += 1
                    weight.append(input_data)
                    CountNode.append(1)
                    adaptiveSig.append(self.SigmaEstimation(DATA, sampleNum, Lambda))
                    edge = self._add_new_node(edge, numNodes)
                else:
                    # Update the winning node
                    CountNode[s1] += 1
                    weight[s1] = weight[s1] + (1 / (10 * CountNode[s1])) * (input_data - weight[s1])

                    if minCIM >= Lcim_s2:
                        # Create an edge between s1 and s2 nodes
                        edge[s1, s2] = 1
                        edge[s2, s1] = 1

                        # Update weight of neighbors of s1 node
                        s1_neighbors = np.where(edge[s1, :] != 0)[0]
                        for k in s1_neighbors:
                            weight[k] = weight[k] + (1 / (100 * CountNode[k])) * (input_data - weight[k])

            # Topology Adjustment: delete isolated nodes based on edge connectivity
            if sampleNum % Lambda == 0:
                edge, weight, CountNode, adaptiveSig, numNodes = self._topology_adjustment(edge, weight, CountNode, adaptiveSig, numNodes)

        # Cluster Labeling based on edge (equivalent to conncomp in MATLAB)
        LabelCluster = self._label_clusters(edge)

        self.numNodes = numNodes
        self.weight = weight
        self.CountNode = CountNode
        self.adaptiveSig = adaptiveSig
        self.LabelCluster = LabelCluster
        self.edge = edge

    def SigmaEstimation(self, DATA, sampleNum, Lambda):
        if DATA.shape[0] < Lambda:
            exNodes = DATA
        elif sampleNum - Lambda <= 0:
            exNodes = DATA[:Lambda, :]
        else:
            exNodes = DATA[(sampleNum + 1) - Lambda:sampleNum, :]

        qStd = np.std(exNodes, axis=0)
        qStd[qStd == 0] = 1.0E-6
        n, d = exNodes.shape
        estSig = np.median(((4 / (2 + d)) ** (1 / (4 + d))) * qStd * n ** (-1 / (4 + d)))
        return estSig

    def CIM(self, X, Y, sig):
        n, att = Y.shape
        g_Kernel = np.zeros((n, att))

        for i in range(att):
            g_Kernel[:, i] = self.GaussKernel(X[i] - Y[:, i], sig)

        ret0 = 1  # This is equivalent to the kernel of 0 in MATLAB
        ret1 = np.mean(g_Kernel, axis=1)
        cim = np.sqrt(ret0 - ret1)
        return cim

    def GaussKernel(self, sub, sig):
        return np.exp(-sub ** 2 / (2 * sig ** 2))

    def _add_new_node(self, edge, numNodes):
        # Increase the size of the edge matrix as needed
        if edge.shape[0] < numNodes:
            new_edge = np.zeros((numNodes, numNodes))
            new_edge[:edge.shape[0], :edge.shape[1]] = edge
            edge = new_edge
        return edge

    def _topology_adjustment(self, edge, weight, CountNode, adaptiveSig, numNodes):
        # Identify and delete isolated nodes
        nNeighbor = np.sum(edge, axis=1)
        deleteNodeEdge = (nNeighbor == 0)

        # Convert to numpy arrays
        weight = np.array(weight)
        CountNode = np.array(CountNode)
        adaptiveSig = np.array(adaptiveSig)

        # Perform boolean indexing
        weight = weight[~deleteNodeEdge]  # Adjust for correct indexing
        CountNode = CountNode[~deleteNodeEdge]
        adaptiveSig = adaptiveSig[~deleteNodeEdge]

        # Adjust edge matrix
        edge = edge[~deleteNodeEdge, :]
        edge = edge[:, ~deleteNodeEdge]

        # Update the number of nodes
        numNodes = np.sum(~deleteNodeEdge)

        return edge, weight.tolist(), CountNode.tolist(), adaptiveSig.tolist(), numNodes

    def _label_clusters(self, edge):
        # Using networkx for graph-based clustering
        G = nx.from_numpy_matrix(edge)
        LabelCluster = list(nx.connected_components(G))
        cluster_labels = np.zeros(edge.shape[0], dtype=int)
        for cluster_id, nodes in enumerate(LabelCluster):
            for node in nodes:
                cluster_labels[node] = cluster_id + 1
        return cluster_labels
