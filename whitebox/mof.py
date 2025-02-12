import ast
import os
import re

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

#これの決定木版も作成する，最大深さは4
def objective_tree(trial, X_train, y_train):
    max_depth = trial.suggest_int("max_depth", 2, 4)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)

    clf = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )

    X_cal, X_val, y_cal, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    clf.fit(X_cal, y_cal)
    y_pred = clf.predict(X_val)
    return accuracy_score(y_val, y_pred)


if __name__ == '__main__':
    # データセットのパス
    data_name = "yeast"
    DATASET_DIR = f"C:/Users/Ayato Tomofuji/Documents/Mof/MoFGBMLPy/dataset/{data_name}"
    MoF_DIR = f"C:/Users/Ayato Tomofuji/Documents/Mof/MoFGBMLPy/results/1/{data_name}"
    RANDOM_SEED = 42

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
    results2 = []
    res_id = 1


    def parse_list_column(s):
        try:
            lst = ast.literal_eval(s)
            return np.array([-1 if x is None else int(x) for x in lst])  # 文字列をリストに変換
        except (ValueError, SyntaxError):
            return []  # 失敗した場合は空リスト

    for train_file, test_file in zip(train_files, test_files):
        train_path = os.path.join(DATASET_DIR, train_file)
        test_path = os.path.join(DATASET_DIR, test_file)

        train_data = np.loadtxt(train_path, delimiter=",", skiprows=1, usecols=range(1, dim + 1))
        test_data = np.loadtxt(test_path, delimiter=",", skiprows=1, usecols=range(1, dim + 1))

        X_train, y_train = train_data[:, :-1], train_data[:, -1]
        X_test, y_test = test_data[:, :-1], test_data[:, -1]
        # 文字列として格納されているリストをリスト型に変換する関数

        identifier = os.path.basename(test_path).split(f"_{data_name}-10tst")[0]
        # Base classifier（MoFGBML）
        df = pd.read_csv(f"{MoF_DIR}/{res_id}/results.csv", delimiter=",")
        #df = pd.read_csv(f"{MoF_DIR}/{identifier}/results.csv", delimiter=",")


        # prediction_train と prediction_test を numpy.ndarray に変換
        df["prediction_train"] = df["prediction_train"].apply(parse_list_column)
        df["prediction_test"] = df["prediction_test"].apply(parse_list_column)
        # **Optunaで最適化されたRandomForestをhard samplesに適用**
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
        study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=20)
        best_params = study.best_params
#
        defe_clf = RandomForestClassifier(**best_params, random_state=RANDOM_SEED, n_jobs=-1)
        defe_clf.fit(X_train, y_train)
        defe_predictions_train = defe_clf.predict(X_train)
        defe_predictions_test = defe_clf.predict(X_test)

        df_filtered = df[df["num_rules"] == 1]  # `num_rules` が num_rule のものを抽出
        base_predictions_train = df_filtered.iloc[0]["prediction_train"]
        # **ユニークな num_rules の値を取得**
        unique_num_rules = sorted(df["num_rules"].unique())  # 昇順にソート


        # **ユニークなルール数ごとにループ**
        for num_rule in unique_num_rules:
            df_filtered = df[df["num_rules"] == num_rule]  # `num_rules` が num_rule のものを抽出
            if df_filtered.empty:
                continue  # 該当するデータがない場合はスキップ

            # **Base classifier（決定木）**
            base_predictions_train = df_filtered.iloc[0]["prediction_train"]
            base_predictions_test = df_filtered.iloc[0]["prediction_test"]
            base_predictions_train = np.where(base_predictions_train == None, -1, base_predictions_train)

            easy_mask_train = base_predictions_train == y_train
            hard_mask_train = ~easy_mask_train



            base_train_score = accuracy_score(y_train, base_predictions_train)
            print(f"num_rules: {num_rule}, Train Score: {base_train_score:.4f}")
            # **Hard/Easy分類器（Grader）**
            y_easy = np.ones_like(y_train)
            y_easy[hard_mask_train] = 0

            if sum(y_easy == 0) > 2:
                smote = SMOTE(random_state=RANDOM_SEED, k_neighbors=2)
                X_resampled, y_resampled = smote.fit_resample(X_train, easy_mask_train)
            else:
                X_resampled, y_resampled = X_train, easy_mask_train
            grader_clf = DecisionTreeClassifier(max_depth=4, random_state=RANDOM_SEED)
            grader_clf.fit(X_resampled, y_resampled)

            #**訓練データでの評価**


            final_predictions_train = base_predictions_train.copy()
            final_predictions_train[hard_mask_train] = defe_predictions_train[hard_mask_train]
            deferral_rate_train = sum(hard_mask_train) / len(easy_mask_train)
            final_accuracy_train = accuracy_score(y_train, final_predictions_train)
            defe_accuracy_train = accuracy_score(y_train, defe_predictions_train)
            defe_accuracy_train_onhard = accuracy_score(y_train[hard_mask_train], defe_predictions_train[hard_mask_train])
            base_accuracy_train = accuracy_score(y_train, base_predictions_train)
            base_accuracy_train_oneasy = accuracy_score(y_train[easy_mask_train], base_predictions_train[easy_mask_train])



            # **テストデータでの評価**
            test_hard_easy = grader_clf.predict(X_test)
            easy_mask_test = test_hard_easy == 1
            hard_mask_test = test_hard_easy == 0
            final_predictions_test = base_predictions_test.copy()
            final_predictions_test[hard_mask_test] = defe_predictions_test[hard_mask_test]
            final_accuracy_test = accuracy_score(y_test, final_predictions_test)
            defe_accuracy_test = accuracy_score(y_test, defe_predictions_test)
            deferral_rate_test = sum(hard_mask_test) / len(easy_mask_test)
            base_accuracy_test = accuracy_score(y_test, base_predictions_test)
            base_accuracy_test_oneasy = accuracy_score(y_test[easy_mask_test], base_predictions_test[easy_mask_test])
            defe_accuracy_test_onhard = accuracy_score(y_test[hard_mask_test], defe_predictions_test[hard_mask_test])

            results.append((train_file, test_file, num_rule, base_train_score, base_accuracy_test, final_accuracy_train, final_accuracy_test, deferral_rate_train, deferral_rate_test))
            print(
                f"num_rules={num_rule}: {train_file} -> Base Train Acc: {base_train_score:.4f}, Base Test Acc: {base_accuracy_test:.4f}, Final Train Acc: {final_accuracy_train:.4f}, Final Test Acc: {final_accuracy_test:.4f}, Deferral Train Rate: {deferral_rate_train:.4f}, Deferral Test Rate: {deferral_rate_test:.4f}"
            )
        res_id += 1

    # 結果出力
    results_df = pd.DataFrame(results,
                              columns=["Train File", "Test File", "Num Rules", "Base Train Accuracy", "Base Test Accuracy", "Final Train Accuracy", "Final Test Accuracy", "Deferral Train Rate", "Deferral Test Rate",
                                       ])
    # **各 num_rules ごとの統計情報を計算**
    summary_df = results_df.groupby("Num Rules").agg(
        Base_Train_Accuracy_Mean=("Base Train Accuracy", "mean"),
        Base_Test_Accuracy_Mean=("Base Test Accuracy", "mean"),
        Final_Train_Accuracy_Mean=("Final Train Accuracy", "mean"),
        Final_Test_Accuracy_Mean=("Final Test Accuracy", "mean"),
        Deferral_Train_Rate_Mean=("Deferral Train Rate", "mean"),
        Deferral_Test_Rate_Mean=("Deferral Test Rate", "mean"),
        Count=("Num Rules", "count")  # 各ルール数の出現回数
    ).reset_index()

    # **結果を表示**
    print("\nSummary of Metrics by Num Rules:")
    print(summary_df)

    # **詳細データの表示**
    print("\nMean of Each Metric for Each Num Rules:")
    print(summary_df[
              ["Num Rules", "Base_Train_Accuracy_Mean", "Base_Test_Accuracy_Mean", "Final_Train_Accuracy_Mean",
               "Final_Test_Accuracy_Mean", "Deferral_Train_Rate_Mean", "Deferral_Test_Rate_Mean"]])

    print("\nCount of Each Num Rules:")
    print(summary_df[["Num Rules", "Count"]])

    #summary of metrics by num rulesを，csvファイルとして保存
    summary_df.to_csv(f"{MoF_DIR}/summary_of_metrics_by_num_rules.csv", index=False)

