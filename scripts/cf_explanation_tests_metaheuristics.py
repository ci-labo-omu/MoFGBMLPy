import copy
import os

import numpy as np

from mofgbmlpy.data.dataset import Dataset

from mofgbmlpy.data.class_label.class_label_basic import ClassLabelBasic
from pymoo.core.population import Population

from mofgbmlpy.data.input import Input
from mofgbmlpy.fuzzy.knowledge.factory.homo_triangle_knowledge_factory_2_3_4_5 import \
    HomoTriangleKnowledgeFactory_2_3_4_5

from mofgbmlpy.fuzzy.rule.consequent.learning.learning_basic import LearningBasic

from mofgbmlpy.data.pattern import Pattern

from mofgbmlpy.data.output import Output
from mofgbmlpy.main.abstract_main import AbstractMain
from mofgbmlpy.main.pittsburgh.pittsburgh_main import PittsburghMain
from mofgbmlpy.explainer.util import remove_duplicates
from mofgbmlpy.explainer.counterfactual_explainer_benchmark import CounterFactualExplainerBenchmark as CFEBenchmark
from mofgbmlpy.explainer.counterfactual_explainer_metaheuristics import CounterFactualExplainerMetaheuristics as CFEMetaheuristics
from mofgbmlpy.explainer.util import append_rule_classifier

def get_config(data_name, min_num_rules=None, num_evals=5000, verbose=True, interpretability_obj="num-rules"):
    args = [
        "--data-name",
        f"{data_name}",
        "--algorithm-id",
        "0",
        "--experiment-id",
        "0",
        "--train-file",
        f"..\\dataset\\{data_name}\\a0_0_{data_name}-10tra.dat",
        "--test-file",
        f"..\\dataset\\{data_name}\\a0_0_{data_name}-10tra.dat",
        "--terminate-evaluation",
        str(num_evals),
        "--no-output-files",
    ]

    if interpretability_obj == "num-rules":
        args.extend(["--objectives", "error-rate", "num-rules"])
    elif interpretability_obj == "total-rule-length":
        args.extend(["--objectives", "error-rate", "total-rule-length"])
    else:
        raise ValueError(f"Unknown interpretability objective: {interpretability_obj}")

    if verbose:
        args.append("--verbose")

    if min_num_rules is not None:
        args.extend(["--min-rule-num", str(min_num_rules)])

    algo_name = AbstractMain.get_algo_name_from_raw_args(args)
    runner = PittsburghMain(HomoTriangleKnowledgeFactory_2_3_4_5, algo_name)
    res = runner.run(args)

    non_dominated_solutions = res.X
    non_dominated_solutions = remove_duplicates(non_dominated_solutions)

    test_path = f"..\\dataset\\{data_name}\\a0_0_{data_name}-10tst.dat"

    learner = LearningBasic(runner.get_train_set())
    train_set = learner.get_training_set()

    test_set = None
    if os.path.exists(test_path):
        test_set = Input.input_data_set_basic(test_path)

    num_classes = train_set.get_num_classes()

    return num_classes, train_set, test_set, non_dominated_solutions, runner


if __name__ == "__main__":
    # for data_name in ["bupa", "pima"]:
    #     out_path = f"..\\cf_results\\saved_solutions\\{data_name}"
    #     os.makedirs(out_path, exist_ok=True)
    #     _, _, test_dataset, non_dominated_solutions, runner = get_config(data_name, num_evals=10000)
    #
    #     results_data = runner.solutions_list_to_dict_array(non_dominated_solutions.flatten())
    #     Output.save_data(results_data, str(os.path.join(out_path, "results.csv")))
    #
    #     non_dominated_solutions = Population.new(X=non_dominated_solutions)
    #     results_csv = runner.get_results_csv(non_dominated_solutions)
    #     results_xml = runner.get_results_xml(non_dominated_solutions)
    #
    #     out_file = str(os.path.join(out_path, "results.csv"))
    #
    #     with open(out_file, "w") as f:
    #         f.write(results_csv)
    #
    #     out_file = str(os.path.join(out_path, "results.xml"))
    #     Output.save_data(results_xml, out_file, pretty_xml=True)
    #
    #
    # raise Exception("STOP")

    # _, _, test_dataset, non_dominated_solutions, _ = get_config("iris")
    # CFEBenchmark.main_plot_single(CFEMetaheuristics, non_dominated_solutions, test_dataset=test_dataset, cl_idx=29, r_idx=0, c_target=0, sol_idx=3)

    # for data_name in ["bupa", "iris", "pima"]:
    #     num_classes, _, test_dataset, non_dominated_solutions, _ = get_config(data_name, min_num_rules=2)
    #     test_name = "min_num_rules_2"
    #
    #     result_path = f"..\\cf_results\\cf_metaheuristics\\{test_name}\\{data_name}"
    #     CFEBenchmark.main_benchmark(CFEMetaheuristics, num_classes, non_dominated_solutions, out_path=result_path, test_dataset=test_dataset, data_name=data_name, test_name=test_name)

    # for min_num_rules in [1]: #[1, 2]:
    #     for data_name in ["bupa", "iris", "pima"]:
    #         num_classes, train_dataset, test_dataset, non_dominated_solutions, _ = get_config(data_name, min_num_rules=min_num_rules)
    #
    #         result_path = f"../cf_results\\cf_metaheuristics\\param_search\\min_num_rules_{min_num_rules}\\mutation_prob\\{data_name}"
    #         CFEBenchmark.param_search(CFEMetaheuristics, num_classes, non_dominated_solutions, result_path, "mutation_prob", test_dataset=test_dataset, data_name=data_name, min_val=0.01, max_val=0.09, num_experiments=9)
    #         CFEBenchmark.param_search(CFEMetaheuristics, num_classes, non_dominated_solutions, result_path, "mutation_prob", test_dataset=test_dataset, data_name=data_name, min_val=0.001, max_val=0.009, num_experiments=9)
    #         CFEBenchmark.param_search(CFEMetaheuristics, num_classes, non_dominated_solutions, result_path, "mutation_prob", test_dataset=test_dataset, data_name=data_name)
    #
    #         for param_name in ["sampling_change_fs_params_prob", "sampling_fs_type_prob", "mutation_fs_type_prob", "sampling_noise_str", "mutated_param_prob", "crossover_prob", "mutation_revert_to_initial_prob", "crossover_prob", "mutation_prob"]:
    #             result_path = f"../cf_results\\cf_metaheuristics\\param_search\\min_num_rules_{min_num_rules}\\{param_name}\\{data_name}"
    #             CFEBenchmark.param_search(CFEMetaheuristics, num_classes, non_dominated_solutions, result_path, param_name, test_dataset=test_dataset, data_name=data_name)
    #
    #         for param_name in ["n_gen", "pop_size"]:
    #             result_path = f"../cf_results\\cf_metaheuristics\\param_search\\min_num_rules_{min_num_rules}\\{param_name}\\{data_name}"
    #             CFEBenchmark.param_search(CFEMetaheuristics, num_classes, non_dominated_solutions, result_path, param_name, test_dataset=test_dataset, data_name=data_name, vals=[10, 25, 50, 100], is_int=True)


    test_configs = {
        # "num_features_no_change_loss": {"objectives": ["confidence_loss", "num_changed_features"]},
        # "num_features_no_change_loss_less_edits": {"objectives": ["confidence_loss", "num_changed_features"], "mutation_revert_to_initial_prob": 0.3, "sampling_change_fs_params_prob": 0.3},
        # "less_edits": {"mutation_revert_to_initial_prob": 0.3, "sampling_change_fs_params_prob": 0.3},
        # "num_features": {"objectives": ["confidence_loss", "change_loss", "num_changed_features"]},
        "classic": {},
        # "X_crowding": {"use_search_space_crowding": True},
        # "error_rate": {"objectives": ["confidence_loss", "change_loss", "train_error_rate"]},
        # "no_fs_type_change": {"mutation_fs_type_prob": 0.0, "sampling_fs_type_prob": 0.0}
    }
    test_names = list(test_configs.keys())

    # for data_name in ["appendicitis", "bal", "bupa", "contraceptive", "haberman", "heart", "iris", "mammographic", "newthyroid", "page-blocks", "phoneme", "pima", "spectfheart", "tae", "wisconsin", "sonar", "magic", "movement_libras"]:
    # for data_name in ["bupa", "iris", "pima"]:
    for data_name in ["movement_libras"]:
        all_tests_already_exist = True
        for test_name in test_names:
            test_path = f"..\\cf_results\\cf_metaheuristics\\{test_name}\\{data_name}"
            if not os.path.exists(test_path):
                all_tests_already_exist = False
                break
        if all_tests_already_exist:
            print(f"All tests on {data_name} have already been run, skipping...")
            continue

        num_classes, _, test_dataset, non_dominated_solutions, _ = get_config(data_name)
        #
        # results_path = f"..{os.sep}saved_solutions{os.sep}{data_name}{os.sep}results.xml"
        # train_path = f"..\\dataset\\{data_name}\\a0_0_{data_name}-10tra.dat"
        # test_path = f"..\\dataset\\{data_name}\\a0_0_{data_name}-10tst.dat"
        # if not os.path.exists(test_path):
        #     test_path = train_path
        # non_dominated_solutions, _, _ = PittsburghMain.import_xml_classifiers(results_path, train_path, test_path, objectives=["error-rate", "num-rules"])
        #
        # print("len non_dominated_solutions before:", len(non_dominated_solutions))
        #
        # non_dominated_solutions = non_dominated_solutions.get("X")

        for test_name, config in test_configs.items():
            result_path = f"..\\cf_results\\cf_metaheuristics\\{test_name}\\{data_name}"
            CFEBenchmark.main_benchmark(CFEMetaheuristics, num_classes, non_dominated_solutions, out_path=result_path, test_dataset=test_dataset, data_name=data_name, test_name=test_name, **config)


    # Example run on a simple dataset
    # _, train_dataset, test_dataset, non_dominated_solutions, _ = get_config("iris")
    # cl_idx = 29  # consequent class is 2
    # cf_rule = CFEBenchmark.main_plot_single(CFEMetaheuristics, non_dominated_solutions, test_dataset=test_dataset, cl_idx=cl_idx, r_idx=0, c_target=0, sol_idx=3)[0]
    #
    # initial_classifier = non_dominated_solutions[cl_idx][0]
    #
    # new_cl = append_rule_classifier(initial_classifier, cf_rule, train_set=train_dataset)
    #
    # non_dominated_solutions = np.array([[new_cl]])
    # CFEBenchmark.main_plot_single(CFEMetaheuristics, non_dominated_solutions, test_dataset=test_dataset, cl_idx=0, r_idx=0, c_target=1, sol_idx=2)

    # Example run on a simple dataset, here we have 1 CF rule per rule
    # _, train_dataset, test_dataset, non_dominated_solutions, _ = get_config("iris_merged_1_2", min_num_rules=2)

    # for i in range(len(non_dominated_solutions)):
    #     cl = non_dominated_solutions[i][0]
    #     if len(cl.get_vars()) == 2 and cl.get_total_rule_length() == 2:
    #         print(i, cl.get_total_rule_length())
    #
    #         for var in cl.get_vars():
    #             print(var.get_rule())
    #         print("====================")
    #
    # cl_idx = 47  # different attributes used, only 1 per rule, 2 rules
    # initial_classifier = non_dominated_solutions[cl_idx][0]

    # for var in initial_classifier.get_vars():
    #     print(var.get_rule())
    #
    # var_names = ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"]
    # class_labels = ["Setosa", "Versicolor or Virginica"]
    # cf_rule_1 = CFEBenchmark.main_plot_single(CFEMetaheuristics, non_dominated_solutions, var_names=var_names, class_labels=class_labels, test_dataset=test_dataset, cl_idx=cl_idx, r_idx=0, c_target=1, sol_idx=6, sampling_fs_type_prob=0.0, mutation_fs_type_prob=0.0, objectives=["confidence_loss", "change_loss"], decision_boundaries_fixed_vals=[0.5, None, None, 0.5])[0]
    #
    # new_cl = append_rule_classifier(initial_classifier, cf_rule_1, train_set=train_dataset)
    # non_dominated_solutions = np.array([[new_cl]])
    #
    # cf_rule_2 = CFEBenchmark.main_plot_single(CFEMetaheuristics, non_dominated_solutions, var_names=var_names, class_labels=class_labels, test_dataset=test_dataset, cl_idx=0, r_idx=1, c_target=0, sol_idx=3, sampling_fs_type_prob=0.0, mutation_fs_type_prob=0.0, objectives=["confidence_loss", "change_loss"], decision_boundaries_fixed_vals=[0.5, None, None, 0.5])[0]

    # Example on Pima
    # _, train_dataset, test_dataset, non_dominated_solutions, _ = get_config("pima", min_num_rules=2, verbose=True, interpretability_obj="total-rule-length")

    # for i in range(len(non_dominated_solutions)):
    #     cl = non_dominated_solutions[i][0]
    #     if len(cl.get_vars()) == 2:
    #         print(i, cl.get_total_rule_length(), cl.get_error_rate())
    #
    #         for var in cl.get_vars():
    #             print(var.get_rule())
    #         print("====================")

    # cl_idx = 5
    # initial_classifier = non_dominated_solutions[cl_idx][0]
    # decision_boundaries_fixed_vals = [0.18, None, 0.59, 0.23, 0.03, 0.48, 0.13, None]

    # for i in range(train_dataset.get_num_dim()):
    #     feature_values = [p.get_attribute_value(i) for p in train_dataset.get_patterns()]
    #     median_value = np.median(feature_values)
    #     print(f"Feature {i} median value: {median_value}")

    # for var in initial_classifier.get_vars():
    #     print(var.get_rule())

    # var_names = ["Preg", "Plas", "Pres", "Skin", "Insu", "Mass", "Pedi", "Age"]
    # class_labels = ["Tested negative", "Tested positive"]
    # cf_rule_1 = CFEBenchmark.main_plot_single(CFEMetaheuristics, non_dominated_solutions, plot=False, var_names=var_names, class_labels=class_labels, test_dataset=test_dataset, cl_idx=cl_idx, r_idx=0, c_target=1, sol_idx=15, sampling_fs_type_prob=0.0, mutation_fs_type_prob=0.0, objectives=["confidence_loss", "change_loss"],decision_boundaries_fixed_vals=decision_boundaries_fixed_vals)[0]
    #
    # new_cl = append_rule_classifier(initial_classifier, cf_rule_1, train_set=train_dataset)
    # non_dominated_solutions = np.array([[new_cl]])
    #
    # cf_rule_2 = CFEBenchmark.main_plot_single(CFEMetaheuristics, non_dominated_solutions, plot=False, var_names=var_names, class_labels=class_labels, test_dataset=test_dataset, cl_idx=0, r_idx=1, c_target=0, sol_idx=39, sampling_fs_type_prob=0.0, mutation_fs_type_prob=0.0, objectives=["confidence_loss", "change_loss"],decision_boundaries_fixed_vals=decision_boundaries_fixed_vals)[0]
    # new_cl = append_rule_classifier(new_cl, cf_rule_2, train_set=train_dataset)
    #
    # rules_names = ["Factual Rule 1", "Factual Rule 2", "CF Rule 1", "CF Rule 2"]
    #
    # new_cl.update_winners_and_errors(train_dataset)
    # new_cl.plot_rules(var_names, rules_names, dims=[1, 7])
    # print(f"Initial classifier error rate: {initial_classifier.get_error_rate()}")
    # print(f"New classifier error rate: {new_cl.get_error_rate()}")
    # for var in new_cl.get_vars():
    #     print(var.get_rule())



