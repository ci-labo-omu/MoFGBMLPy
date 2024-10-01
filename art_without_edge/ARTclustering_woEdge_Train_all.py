import numpy as np


class ARTNet_all:
    def __init__(self, numNodes=0, weight=None, LabelCluster=None, CountNode=None, adaptiveSig=None, Lambda=50,
                 minCIM=0.15):
        if weight is None:
            weight = []
        if CountNode is None:
            # CountNodeをクラスごとの勝利回数を保存する辞書形式に変更
            CountNode = [{} for _ in range(numNodes)]
        if adaptiveSig is None:
            adaptiveSig = []
        if LabelCluster is None:
            LabelCluster = []

        self.numNodes = numNodes
        self.weight = weight  # 各ノードの座標
        self.CountNode = CountNode  # 各ノードごとのクラス別カウント
        self.adaptiveSig = adaptiveSig
        self.Lambda = Lambda
        self.minCIM = minCIM
        self.LabelCluster = LabelCluster  # ノードごとのクラス情報を管理するリスト

    def ARTclustering_woEdge_Train(self, DATA, LABELS):
        numNodes = self.numNodes
        weight = self.weight
        CountNode = self.CountNode
        adaptiveSig = self.adaptiveSig
        Lambda = self.Lambda
        minCIM = self.minCIM

        for sampleNum in range(DATA.shape[0]):
            if len(weight) == 0 or sampleNum % Lambda == 0:
                estSigCA = self.SigmaEstimation(DATA, sampleNum, Lambda)

            input_data = DATA[sampleNum, :]
            input_label = LABELS[sampleNum]  # 現在のデータ点のクラスラベル

            if len(weight) < 1:
                # 新しいノードの追加
                numNodes += 1
                weight.append(input_data)
                CountNode.append({input_label: 1})  # クラス情報を辞書形式で記録
                adaptiveSig.append(estSigCA)
            else:
                globalCIM = self.CIM(input_data, np.array(weight), np.mean(adaptiveSig))
                gCIM = globalCIM

                Lcim_s1, s1 = np.min(gCIM), np.argmin(gCIM)
                gCIM[s1] = np.inf
                Lcim_s2, s2 = np.min(gCIM), np.argmin(gCIM)

                if minCIM < Lcim_s1:
                    # 新しいノードの追加
                    numNodes += 1
                    weight.append(input_data)
                    CountNode.append({input_label: 1})  # クラスごとに辞書でカウント
                    adaptiveSig.append(self.SigmaEstimation(DATA, sampleNum, Lambda))
                else:
                    # 既存のノードにデータ点を追加
                    if input_label in CountNode[s1]:
                        CountNode[s1][input_label] += 1  # クラスの勝利回数を更新
                    else:
                        CountNode[s1][input_label] = 1  # 新しいクラスのカウントを追加

                    weight[s1] = weight[s1] + (1 / (10 * sum(CountNode[s1].values()))) * (input_data - weight[s1])

                    if minCIM >= Lcim_s2:
                        if input_label in CountNode[s2]:
                            CountNode[s2][input_label] += 1  # 二番目に近いノードに追加
                        else:
                            CountNode[s2][input_label] = 1
                        weight[s2] = weight[s2] + (1 / (100 * sum(CountNode[s2].values()))) * (input_data - weight[s2])

        self.numNodes = numNodes
        self.weight = weight
        self.CountNode = CountNode
        self.adaptiveSig = adaptiveSig
        self.LabelCluster = [1] * len(weight)  # 仮に1を設定（適宜変更可能）

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

    def display_class_count(self):
        # 各ノードごとにクラスごとの勝利回数を表示する
        for i, count_dict in enumerate(self.CountNode):
            print(f"Node {i}: {count_dict}")
