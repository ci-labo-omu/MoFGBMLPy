import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
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

def objective_base(trial, X_train, y_train):
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



RANDOM_SEED = 42
if __name__ == '__main__':
    # データセットのパス
    data_name = "cancer"
    DATASET_DIR = f"C:/Users/Ayato Tomofuji/Documents/Mof/MoFGBMLPy/dataset/{data_name}"
    TARGET_FOLDS = ["a0", "a1"]

    all_files = sorted(os.listdir(DATASET_DIR))
    train_files = [f for f in all_files if any(f.startswith(fold) and "tra" in f for fold in TARGET_FOLDS)]
    test_files = [f.replace("tra", "tst") for f in train_files]
    optuna.logging.disable_default_handler()
    with open(f"{DATASET_DIR}/{train_files[0]}", "r") as f:
        header = f.readline().strip().split(",")
        dim = int(header[1])

    results = []
    importances = []

    for train_file, test_file in zip(train_files, test_files):
        train_path = os.path.join(DATASET_DIR, train_file)
        test_path = os.path.join(DATASET_DIR, test_file)
        #train_pathから，a0_0という部分を取り出す
        identifier = os.path.basename(test_path).split(f"_{data_name}-10tst")[0]

        train_data = np.loadtxt(train_path, delimiter=",", skiprows=1, usecols=range(1, dim + 1))
        test_data = np.loadtxt(test_path, delimiter=",", skiprows=1, usecols=range(1, dim + 1))

        X_train, y_train = train_data[:, :-1], train_data[:, -1]
        X_test, y_test = test_data[:, :-1], test_data[:, -1]

        study_base = optuna.create_study(direction="maximize")
        study_base.optimize(lambda trial: objective_base(trial, X_train, y_train), n_trials=50)
        # Base classifier（決定木）
        best_params_base = study_base.best_params
        base_clf = DecisionTreeClassifier(**best_params_base, random_state=42)
        base_clf.fit(X_train, y_train)
        base_predictions_train = base_clf.predict(X_train)
        # Easyなところのマスク
        easy_mask_train = base_predictions_train == y_train
        hard_mask_train = ~easy_mask_train
        base_accuracy_train = accuracy_score(y_train, base_predictions_train)
        base_accuracy_train_easy = accuracy_score(y_train[easy_mask_train], base_predictions_train[easy_mask_train])


        # Optunaで最適化されたRandomForestをhard samplesに適用
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=50)
        best_params = study.best_params
        defe_clf = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
        defe_clf.fit(X_train, y_train)
        print(defe_clf.score(X_train, y_train))
        defe_predictions_train = defe_clf.predict(X_train)
        deferral_rate_train = sum(hard_mask_train) / len(easy_mask_train)
        defe_accuracy_train = accuracy_score(y_train, defe_predictions_train)
        defe_accuracy_train_hard = accuracy_score(y_train[hard_mask_train], defe_predictions_train[hard_mask_train])
        final_predictions_train = base_predictions_train.copy()
        final_predictions_train[hard_mask_train] = defe_predictions_train[hard_mask_train]
        final_accuracy_train = accuracy_score(y_train, final_predictions_train)
        # Hard/Easy分類器（Grader）
        y_easy = np.ones_like(y_train)
        y_easy[hard_mask_train] = 0
        if sum(y_easy == 0) > 2:
            smote = SMOTE(random_state=RANDOM_SEED, k_neighbors=2)
            X_resampled, y_resampled = smote.fit_resample(X_train, easy_mask_train)
        else:
            X_resampled, y_resampled = X_train, easy_mask_train
        grader_clf = DecisionTreeClassifier(max_depth=4, random_state=RANDOM_SEED)
        grader_clf.fit(X_resampled, y_resampled)

        # テストデータでの評価
        test_hard_easy = grader_clf.predict(X_test)
        easy_mask_test = test_hard_easy == 1
        hard_mask_test = test_hard_easy == 0
        base_predictions_test = base_clf.predict(X_test)
        defe_predictions_test = defe_clf.predict(X_test)

        final_predictions_test = base_predictions_test.copy()
        final_predictions_test[hard_mask_test] = defe_predictions_test[hard_mask_test]

        importances.append(defe_clf.feature_importances_)
        base_accuracy_test = accuracy_score(y_test, base_predictions_test)
        base_accuracy_test_easy = accuracy_score(y_test[easy_mask_test], base_predictions_test[easy_mask_test])
        final_accuracy_test = accuracy_score(y_test, final_predictions_test)
        defe_accuracy = accuracy_score(y_test, defe_predictions_test)
        defe_accuracy_test = accuracy_score(y_test, defe_predictions_test)
        defe_accuracy_test_hard = accuracy_score(y_test[hard_mask_test], defe_predictions_test[hard_mask_test])

        deferral_rate_test = sum(hard_mask_test) / len(easy_mask_test)
        results.append((train_file, test_file, base_accuracy_train, base_accuracy_test,
                        final_accuracy_train, final_accuracy_test, deferral_rate_train,
                        deferral_rate_test,
                        base_accuracy_train_easy,  # ここから追記
                        base_accuracy_test_easy, defe_accuracy_train, defe_accuracy_train_hard,
                        defe_accuracy_test, defe_accuracy_test_hard))
        print(
            f"Base Train Acc: {base_accuracy_train:.4f}, Base Test Acc: {base_accuracy_test:.4f}, Final Train Acc: {final_accuracy_train:.4f}, Final Test Acc: {final_accuracy_test:.4f}, Deferral Train Rate: {deferral_rate_train:.4f}, Deferral Test Rate: {deferral_rate_test:.4f}")
    # 結果をdfに、ここに全部まとめる
    results_df = pd.DataFrame(results,
                              columns=["Train File", "Test File", "Base Train Accuracy",
                                       "Base Test Accuracy", "Final Train Accuracy", "Final Test Accuracy",
                                       "Deferral Train Rate", "Deferral Test Rate", "Base Train Accuracy on Easy",
                                       "Base Test Accuracy on Easy", "Deferral Train Accuracy",
                                       "Deferral Train Accuracy on Hard", "Deferral Test Accuracy",
                                       "Deferral Test Accuracy on Hard",
                                       ])
    # **各 num_rules ごとの統計情報を計算**
    summary_df1 = results_df[["Final Train Accuracy", "Final Test Accuracy", "Deferral Train Rate", "Deferral Test Rate"]]

    summary_df2 = results_df[["Base Train Accuracy", "Base Test Accuracy",
                              "Base Train Accuracy on Easy", "Base Test Accuracy on Easy",
                              "Deferral Train Accuracy", "Deferral Test Accuracy",
                              "Deferral Train Accuracy on Hard", "Deferral Test Accuracy on Hard"]]

    mean_importances = np.mean(importances, axis=0)
    mdi_importances = pd.Series(mean_importances, index=None).sort_values(ascending=True)
    ax = mdi_importances.plot.barh()
    ax.set_title(f"Random Forest Feature Importances: {data_name}")
    ax.figure.tight_layout()
    plt.show()
    # **結果を表示**
    print("\nSummary of Metrics")
    print(summary_df1)
    # **詳細データの表示**

    # summary of metrics by num rulesを，csvファイルとして保存
    summary_df1.to_csv(f"whitebox/summary_of_metrics_{data_name}1.csv", index=False)
    summary_df2.to_csv(f"whitebox/summary_of_metrics_{data_name}2.csv", index=False)