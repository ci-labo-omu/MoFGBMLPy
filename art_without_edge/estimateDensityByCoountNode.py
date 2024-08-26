import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def estimateDensityByCountNode(net):
    # ノード位置とカウントを取得
    node_positions = np.array(net.weight)
    count_node = np.array(net.CountNode)
    # Silverman's Ruleに基づくバンド幅の計算
    print(node_positions)
    print(count_node)
    #ノードの座標と，各ノードのカウントをファイルに書き出す．各ノードの座標の後ろにカウントを書く，4次元ベクトルのリストで
    #書き出す．
    with open('node_positions.csv', 'w') as f:
        for i in range(len(node_positions)):
            f.write(str(node_positions[i][0]) + ', ' + str(node_positions[i][1]) + ', ' + str(count_node[i]) + '\n')

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
            kernel_values = count_node * np.exp(-(distances_x ** 2 + distances_y ** 2) / 2)

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
