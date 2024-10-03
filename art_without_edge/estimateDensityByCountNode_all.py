import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def convert_dicts_to_lists(dict_array):
    result = []
    for d in dict_array:
        # 各クラスの寄与数を初期化
        class_list = [0, 0, 0, 0]  # クラス0からクラス3までの寄与数
        for key, value in d.items():
            class_list[key] = value  # 辞書のキーに対応するクラスに寄与数を挿入
        result.append(class_list)  # リストを結果に追加
    return result
def estimateDensityByCountNode_all(net, data):
    # ノード位置とカウントを取得
    node_positions = np.array(net.weight)[:,:-1]
    print(node_positions)
    count_node = np.array(net.CountNode)
    #辞書をリストに変換，[0, 1, 2, 3]の順番で寄与数を書いている
    counts_list = convert_dicts_to_lists(count_node)
    #ノードの座標と，各ノードのカウントをファイルに書き出す．各ノードの座標の後ろにカウントを書く，4次元ベクトルのリストで
    #書き出す．
    print(counts_list)
    print(node_positions)
    # ノードの座標とカウントを結合したデータを作成
    data = np.hstack([node_positions, np.array(counts_list)])

    # 書き出し部分
    with open('node_positions_all/all_4dim_50_070.csv', 'a') as f:
        np.savetxt(f, data, delimiter=', ', fmt='%s')

    exit()
    #各ノードの，各クラスからの勝利回数を表した辞書
    counts = [sum(count_node[i].values()) for i in range(len(count_node))]



    n = node_positions.shape[0]
    sigma_x = np.std(node_positions[:, 0])
    sigma_y = np.std(node_positions[:, 1])

    h_x = sigma_x * (4 / (3 * n)) ** (1 / 5)
    h_y = sigma_y * (4 / (3 * n)) ** (1 / 5)

    # グリッドの範囲と解像度の設定
    grid_size = 100  # グリッドの解像度
    x_grid = np.linspace(0, 1, grid_size)
    y_grid = np.linspace(0, 1, grid_size)

    X, Y = np.meshgrid(x_grid, y_grid)
    density = np.zeros_like(X)

    # 各グリッドポイントの密度を計算
    for i in range(grid_size):
        for j in range(grid_size):
            # グリッドポイントの座標
            grid_point = np.array([X[i, j], Y[i, j]])

            # 各ノードに対するカーネル密度の計算
            distances_x = (node_positions[:, 0] - grid_point[0]) / h_x
            distances_y = (node_positions[:, 1] - grid_point[1]) / h_y
            kernel_values = counts * np.exp(-(distances_x ** 2 + distances_y ** 2) / 2)

            # 密度を累積
            density[i, j] = np.sum(kernel_values)

    # 密度を[0, 1]の範囲に正規化
    density /= np.max(density)

    # 3Dサーフェスプロット
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, density, cmap='jet', edgecolor='none', alpha=0.8)

    # カラーバーの設定
    cbar = plt.colorbar(surf, ax=ax)
    surf.set_clim(0, 1)
    cbar.set_ticks(np.arange(0, 1.1, 0.1))
    cbar.set_ticklabels([f'{x:.1f}' for x in np.arange(0, 1.1, 0.1)])

    # 軸の設定
    ax.set_xticks(np.arange(0.0, 1.1, 0.2))
    ax.set_yticks(np.arange(0.0, 1.1, 0.2))
    ax.set_zticks(np.arange(0.0, 1.1, 0.2))

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_zlim([0.0, 1.0])
    # xとyのaxの向きを逆にする
    ax.invert_xaxis()
    ax.invert_yaxis()

    ax.set_title('Estimated Density by CountNode', fontsize=14)
    ax.set_xlabel('X', fontsize=14)
    ax.set_ylabel('Y', fontsize=14)
    ax.set_zlabel('Density', fontsize=14)
    ax.view_init(elev=30, azim=45)  # 3Dビューを設定

    plt.show()
