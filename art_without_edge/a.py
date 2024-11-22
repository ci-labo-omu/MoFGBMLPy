import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from imblearn.under_sampling import CondensedNearestNeighbour, TomekLinks

if __name__ == '__main__':
    num_features = 4
    X, y = make_classification(n_samples=8000,  # サンプル数
                                  n_features=num_features,  # 特徴量の数（2つの特徴量）
                                  flip_y=0,
                                  class_sep=2.2,
                                  n_informative=num_features,  # 有益な特徴量の数
                                  n_redundant=0,  # 冗長な特徴量の数
                                  n_clusters_per_class=1,  # クラスごとのクラスター数
                                  n_classes=4,  # クラス数（4クラス分類）
                                  random_state=42)
    X = MinMaxScaler().fit_transform(X)
    #Xとyを，分かりやすいように横に並べて表示
    data = np.hstack([X, y.reshape(-1, 1)])

    if num_features == 4:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # 散布図を2変数間でプロット
        plot_idx = 0
        for i in range(4):
            for j in range(i + 1, 4):  # i < j として、全てのペア (i, j) を作成
                ax = axes.ravel()[plot_idx]  # axesのインデックスを取得
                for class_value in np.unique(y):  # クラスごとにプロット
                    indices = np.where(y == class_value)
                    ax.scatter(X[indices, i], X[indices, j], label=f'Class {class_value + 1}', s=50, alpha=0.6)

                ax.set_xlim(-0.05, 1.05)
                ax.set_ylim(-0.05, 1.05)
                ax.set_xlabel(f'x{i + 1}')
                ax.set_ylabel(f'x{j + 1}')
                ax.legend()

                plot_idx += 1  # 次のプロット位置へ

        plt.show()



    if num_features == 200:
        # 2D plot
        fig, ax = plt.subplots()
        for class_value in range(4):
            indices = np.where(y == class_value)
            ax.scatter(X[indices, 0], X[indices, 1], label=f'Class {class_value + 1}', s=50, alpha=0.6)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.legend()

    if num_features == 300:
        # 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for class_value in range(4):
            indices = np.where(y == class_value)
            ax.scatter(X[indices, 0], X[indices, 1], X[indices, 2], label=f'Class {class_value + 1}', s=50, alpha=0.6)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_zlim(-0.05, 1.05)
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel('x3')  # Z軸のラベルを追加
        ax.legend()

    clf = DecisionTreeClassifier(max_depth=4)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    clf.fit(X_train, y_train)
    print(f"8000データでテスト{clf.score(X_test, y_test)}")
    #決定木をプロット
    from sklearn.tree import plot_tree
    plt.figure(figsize=(10, 10))
    #plot_tree(clf, filled=True)

    with open('node_positions/node_positions4dim_50_75.csv', 'r') as f:
        #ヘッダ行には行数，次元数，クラス数が記載されている．最後の列は読む必要がない．これをデータセットとしたい．
        #Xにデータ点，yにクラスラベルを格納する．
        #ヘッダ行を読み込む
        header = f.readline().strip().split(',')
        num_rows = int(header[0])
        num_dims = int(header[1])
        num_classes = int(header[2])
        print(num_rows, num_dims, num_classes)
        X_node = np.zeros((num_rows, num_dims))
        y_node = np.zeros(num_rows)
        for i, line in enumerate(f):
            data = line.strip().split(',')[:-1]
            X_node[i] = np.array(data[:-1], dtype=float)
            y_node[i] = data[-1]

    #X_node_train, X_node_test, y_node_train, y_node_test = train_test_split(X_node, y_node, test_size=0.2, random_state=42)
    clf = DecisionTreeClassifier(max_depth=4)
    clf.fit(X_node, y_node)
    print(f"{num_rows}ノードに圧縮")
    print(clf.score(X, y))
    print(len(clf.predict(X)))

    cnn = TomekLinks()
    X_res, y_res = cnn.fit_resample(X, y)
    print(X_res, y_res)
    clf.fit(X_res, y_res)
    print(f"{len(y_res)}ノードに圧縮")
    print(clf.score(X, y))

