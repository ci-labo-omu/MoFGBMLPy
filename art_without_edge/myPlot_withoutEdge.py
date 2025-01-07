import numpy as np
from matplotlib import pyplot as plt


def myPlot_withoutEdge(DATA, net):
    plt.clf()  # Clear the current figure

    w = np.array(net.weight)
    label = np.array(net.LabelCluster)
    N = np.array(net.CountNode)
    num_colors = 10  # Number of colors in the palette

    # データポイントをプロット
    plt.figure(figsize=(10, 8))
    plt.plot(DATA[:, 0], DATA[:, 1], 'o', markeredgecolor=[0.8, 0.8, 0.8], markerfacecolor=[0.7, 0.7, 0.7],
             markersize=3)

    # ノードを描画
    color = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [0.85, 0.325, 0.098],
        [0.929, 0.694, 0.125],
        [0.494, 0.184, 0.556],
        [0.466, 0.674, 0.188],
        [0.301, 0.745, 0.933],
        [0.635, 0.078, 0.184]
    ])
    m = len(color)

    for k in range(len(N)):
        plt.plot(w[k, 0], w[k, 1], '.', color=color[(label[k] - 1) % m], markersize=30)

    # CountNodeを各ノードの右上に描画
    for i in range(len(N)):
        count_str = str(net.CountNode[i])
        plt.text(w[i, 0] + 0.01, w[i, 1] + 0.01, count_str, color='k', fontsize=12)


    plt.xlabel('X', fontsize=14)
    plt.ylabel('Y', fontsize=14)
    plt.grid(True)
    plt.box(True)
    plt.axis('equal')
    plt.xlim = (-0.05, 1.05)
    plt.ylim = (-0.05, 1.05)
    plt.savefig(f'node_classwise.png')
    plt.show()

def myPlot_color(DATA, net):
    #DATAのクラスごとに色分けする,クラスは各行の末尾に書いてある
    plt.clf()  # Clear the current figure
    plt.figure(figsize=(10, 8))
    plt.plot(DATA[:, 0], DATA[:, 1], 'o', markeredgecolor=[0.8, 0.8, 0.8], markerfacecolor=[0.7, 0.7, 0.7],
             markersize=3)
    # カラーマップを取得（例: tab10）
    cmap = plt.get_cmap('tab10')  # 他に 'tab20', 'viridis', 'plasma' なども使用可
    num_colors = cmap.N  # カラーマップ内の色の数

    plt.clf()  # Clear the current figure

    # データポイントをプロット（カラーマップを利用）
    plt.figure(figsize=(10, 8))
    for cls in np.unique(DATA[:, -1]):  # クラスごとのループ
        cls_indices = DATA[:, -1] == cls  # 該当クラスのインデックス
        cls_color = cmap(int(cls) % num_colors)  # クラスにカラーマップの色を割り当て
        plt.scatter(DATA[cls_indices, 0], DATA[cls_indices, 1], c=[cls_color],
                    label=f'Class {int(cls)}', s=40)  # 輪郭を追加

    # ノードを描画（赤色で固定）
    w = np.array(net.weight)
    for k in range(len(net.CountNode)):
        plt.plot(w[k, 0], w[k, 1], '.', color='pink', markersize=30, markeredgecolor='k')  # ノードは赤

    # CountNodeを各ノードの右上に描画
    for i in range(len(net.CountNode)):
        count_str = str(net.CountNode[i])
        plt.text(w[i, 0] + 0.01, w[i, 1] + 0.01, count_str, color='k', fontsize=12)

    # 軸とプロット設定
    plt.xlabel('X', fontsize=14)
    plt.ylabel('Y', fontsize=14)
    plt.legend(fontsize=14)  # 凡例を追加
    plt.grid(True)
    plt.box(True)
    plt.axis('equal')
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.savefig('result_classwise.png')
    plt.show()