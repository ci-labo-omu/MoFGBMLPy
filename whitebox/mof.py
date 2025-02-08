import ast
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
import optuna




# Optunaでハイパーパラメータ最適化
def objective(trial, X_train, y_train):
    n_estimators = trial.suggest_int("n_estimators", 60, 100)
    max_depth = trial.suggest_int("max_depth", 10, 30)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
        n_jobs=-1
    )

    X_cal, X_val, y_cal, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    clf.fit(X_cal, y_cal)
    y_pred = clf.predict(X_val)
    return accuracy_score(y_val, y_pred)


if __name__ == '__main__':
    # データセットのパス
    data_name = "vehicle"
    DATASET_DIR = f"C:/Users/Ayato Tomofuji/Documents/Mof/MoFGBMLPy/dataset/{data_name}"
    MoF_DIR = f"C:/Users/Ayato Tomofuji/Documents/Mof/MoFGBMLPy/results/1/{data_name}"

    TARGET_FOLDS = ["a0", "a1"]
    all_files = sorted(os.listdir(DATASET_DIR))
    train_files = [f for f in all_files if any(f.startswith(fold) and "tra" in f for fold in TARGET_FOLDS)]
    test_files = [f.replace("tra", "tst") for f in train_files]
    optuna.logging.disable_default_handler()
    with open(f"{DATASET_DIR}/{train_files[0]}", "r") as f:
        header = f.readline().strip().split(",")
        dim = int(header[1])
    mode = 0
    results = []
    res_id = 1
    for train_file, test_file in zip(train_files, test_files):
        train_path = os.path.join(DATASET_DIR, train_file)
        test_path = os.path.join(DATASET_DIR, test_file)

        train_data = np.loadtxt(train_path, delimiter=",", skiprows=1, usecols=range(1, dim + 1))
        test_data = np.loadtxt(test_path, delimiter=",", skiprows=1, usecols=range(1, dim + 1))

        X_train, y_train = train_data[:, :-1], train_data[:, -1]
        X_test, y_test = test_data[:, :-1], test_data[:, -1]

        # Base classifier（決定木）
        df = pd.read_csv(f"{MoF_DIR}/{res_id}/results.csv", delimiter=",")


        # 文字列として格納されているリストをリスト型に変換する関数
        def parse_list_column(s):
            try:
                lst = ast.literal_eval(s)
                return np.array([-1 if x is None else int(x) for x in lst])  # 文字列をリストに変換
            except (ValueError, SyntaxError):
                return []  # 失敗した場合は空リスト


        # prediction_train と prediction_test を numpy.ndarray に変換
        df["prediction_train"] = df["prediction_train"].apply(parse_list_column)
        df["prediction_test"] = df["prediction_test"].apply(parse_list_column)



        # Easyなところのマスク
        base_predictions_train = df.loc[0, "prediction_train"]
        base_predictions_test = df.loc[0, "prediction_test"]
        base_predictions_train = np.where(base_predictions_train == None, -1, base_predictions_train)

        easy_mask_train = base_predictions_train == y_train

        hard_mask_train = ~easy_mask_train
        # Optunaで最適化されたRandomForestをhard samplesに適用
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=20)
        best_params = study.best_params
        defe_clf = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
        defe_clf.fit(X_train, y_train)
        print(defe_clf.score(X_train, y_train))
        # Hard/Easy分類器（Grader）
        y_easy = np.ones_like(y_train)
        y_easy[hard_mask_train] = 0
        if sum(y_easy == 0) > 2:
            smote = SMOTE(random_state=42, k_neighbors=2)
            X_resampled, y_resampled = smote.fit_resample(X_train, easy_mask_train)
        else:
            X_resampled, y_resampled = X_train, easy_mask_train

        grader_clf = DecisionTreeClassifier(max_depth=4)
        grader_clf.fit(X_resampled, y_resampled)

        # テストデータでの評価
        test_hard_easy = grader_clf.predict(X_test)
        easy_mask_test = test_hard_easy == 1
        hard_mask_test = test_hard_easy == 0
        defe_predictions_test = defe_clf.predict(X_test)

        final_predictions_test = base_predictions_test.copy()
        final_predictions_test[hard_mask_test] = defe_predictions_test[hard_mask_test]

        base_accuracy = accuracy_score(y_test, base_predictions_test)
        final_accuracy = accuracy_score(y_test, final_predictions_test)
        defe_accuracy = accuracy_score(y_test, defe_predictions_test)
        easy_rate = sum(easy_mask_test) / len(easy_mask_test)
        results.append((train_file, test_file, base_accuracy, easy_rate, defe_accuracy, final_accuracy))
        print(
            f"{train_file} -> Base Acc: {base_accuracy:.4f}, Deferral Acc: {defe_accuracy}, Easy Rate: {easy_rate}, Final Acc: {final_accuracy:.4f},")
        res_id += 1

    # 結果出力
    results_df = pd.DataFrame(results,
                              columns=["Train File", "Test File", "Base Accuracy", "Deferral Acc", "Easy Rate",
                                       "Final Accuracy"])
    print(results_df)
    # final_accuracyの20回平均をとる
    print(results_df["Final Accuracy"].mean())


