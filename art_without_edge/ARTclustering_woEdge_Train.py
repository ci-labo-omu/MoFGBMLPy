import numpy as np


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

        self.numNodes = numNodes
        self.weight = weight
        self.CountNode = CountNode
        self.adaptiveSig = adaptiveSig
        self.Lambda = Lambda
        self.minCIM = minCIM
        self.LabelCluster = LabelCluster

    def ARTclustering_woEdge_Train(self, DATA):
        numNodes = self.numNodes
        weight = self.weight
        CountNode = self.CountNode
        adaptiveSig = self.adaptiveSig
        Lambda = self.Lambda
        minCIM = self.minCIM
        print(DATA.shape)
        for sampleNum in range(DATA.shape[0]):
            if len(weight) == 0 or sampleNum % Lambda == 0:
                estSigCA = self.SigmaEstimation(DATA, sampleNum, Lambda)

            input_data = DATA[sampleNum, :]

            if len(weight) < 1:
                numNodes += 1
                weight.append(input_data)
                CountNode.append(1)
                adaptiveSig.append(estSigCA)
            else:
                globalCIM = self.CIM(input_data, np.array(weight), np.mean(adaptiveSig))
                gCIM = globalCIM

                Lcim_s1, s1 = np.min(gCIM), np.argmin(gCIM)
                gCIM[s1] = np.inf
                Lcim_s2, s2 = np.min(gCIM), np.argmin(gCIM)

                if minCIM < Lcim_s1:
                    numNodes += 1
                    weight.append(input_data)
                    CountNode.append(1)
                    adaptiveSig.append(self.SigmaEstimation(DATA, sampleNum, Lambda))
                else:
                    CountNode[s1] += 1
                    weight[s1] = weight[s1] + (1 / (10 * CountNode[s1])) * (input_data - weight[s1])

                    if minCIM >= Lcim_s2:
                        weight[s2] = weight[s2] + (1 / (100 * CountNode[s2])) * (input_data - weight[s2])

        self.numNodes = numNodes
        self.weight = weight
        self.CountNode = CountNode
        self.adaptiveSig = adaptiveSig
        self.LabelCluster = [1] * len(weight)

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

        ret0 = 1
        ret1 = np.mean(g_Kernel, axis=1)

        cim = np.sqrt(ret0 - ret1)
        return cim

    def GaussKernel(self, sub, sig):
        return np.exp(-sub ** 2 / (2 * sig ** 2))
