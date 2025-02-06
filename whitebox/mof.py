from mofgbmlpy.fuzzy.knowledge.factory.homo_triangle_knowledge_factory_2_3_4_5 import \
    HomoTriangleKnowledgeFactory_2_3_4_5

from mofgbmlpy.main.nsgaii.mofgbml_nsgaii_main import MoFGBMLNSGAIIMain

if __name__ == "__main__":
    args = [
        "--algorithm-id", "1",
        "--experiment-id", "2",
        "--train-file", "None",
        "--test-file", "None",
        "--data-name", "yeast",
        "--terminate-evaluation", "120000",
        "--objectives", "num-rules", "error-rate",
        # "--crossover-type", "pittsburgh-crossover",
        # "--antecedent-factory", "all-combination-antecedent-factory",
        "--crossover-type", "hybrid-gbml-crossover",
        "--verbose",
    ]
    runner = MoFGBMLNSGAIIMain(HomoTriangleKnowledgeFactory_2_3_4_5)
    results = runner.main(args, train=train_set, test=test_set)
    Xs = results.opt.get("X")[:, 0]

    if __name__ == '__main__':
        # データセットのパス
        DATASET_DIR = "C:/Users/Ayato Tomofuji/Documents/Mof/MoFGBMLPy/dataset/satimage"
        TARGET_FOLDS = ["a0", "a1"]

        all_files = sorted(os.listdir(DATASET_DIR))
        train_files = [f for f in all_files if any(f.startswith(fold) and "tra" in f for fold in TARGET_FOLDS)]
        test_files = [f.replace("tra", "tst") for f in train_files]
        optuna.logging.disable_default_handler()
        with open(f"{DATASET_DIR}/{train_files[0]}", "r") as f:
            header = f.readline().strip().split(",")
            dim = int(header[1])

        results = []
        for train_file, test_file in zip(train_files, test_files):
            train_path = os.path.join(DATASET_DIR, train_file)
            test_path = os.path.join(DATASET_DIR, test_file)

            train_data = np.loadtxt(train_path, delimiter=",", skiprows=1, usecols=range(1, dim + 1))
            test_data = np.loadtxt(test_path, delimiter=",", skiprows=1, usecols=range(1, dim + 1))

            X_train, y_train = train_data[:, :-1], train_data[:, -1]
            X_test, y_test = test_data[:, :-1], test_data[:, -1]

            # Base classifier（決定木）
            base_clf = DecisionTreeClassifier(max_depth=4)
            base_clf.fit(X_train, y_train)

            base_predictions_train = base_clf.predict(X_train)
            # Easyなところのマスク
            easy_mask_train = base_predictions_train == y_train
            hard_mask_train = ~easy_mask_train
            # Optunaで最適化されたRandomForestをhard samplesに適用
            study = optuna.create_study(direction="maximize")
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
            base_predictions_test = base_clf.predict(X_test)
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

        # 結果出力
        results_df = pd.DataFrame(results,
                                  columns=["Train File", "Test File", "Base Accuracy", "Deferral Acc", "Easy Rate",
                                           "Final Accuracy"])
        print(results_df)
        # final_accuracyの20回平均をとる
        print(results_df["Final Accuracy"].mean())