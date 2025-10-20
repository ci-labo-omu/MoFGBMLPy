import os

from mofgbmlpy.data.input import Input
from mofgbmlpy.fuzzy.knowledge.factory.homo_triangle_knowledge_factory_2_3_4_5 import \
    HomoTriangleKnowledgeFactory_2_3_4_5

from mofgbmlpy.fuzzy.rule.consequent.learning.learning_basic import LearningBasic
from mofgbmlpy.main.abstract_main import AbstractMain
from mofgbmlpy.main.pittsburgh.pittsburgh_main import PittsburghMain
from mofgbmlpy.explainer.util import remove_duplicates
from mofgbmlpy.explainer.counterfactual_explainer_benchmark import CounterFactualExplainerBenchmark as CFEBenchmark
from mofgbmlpy.explainer.counterfactual_explainer_metaheuristics import CounterFactualExplainerMetaheuristics as CFEMetaheuristics


def get_config(data_name, min_num_rules=None, num_evals=5000):
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
        "--objectives",
        "error-rate",
        "num-rules",
    ]

    if min_num_rules is not None:
        args.extend(["--min-num-rules", str(min_num_rules)])

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

    return num_classes, train_set, test_set, non_dominated_solutions

if __name__ == "__main__":
    # _, _, _, non_dominated_solutions = get_config("bupa")
    # CFEBenchmark.main_plot_single(CFEMetaheuristics, non_dominated_solutions)

    # for data_name in ["bupa", "iris", "pima"]:
    #
    #     num_classes, _, test_dataset, non_dominated_solutions = get_config(data_name, min_num_rules=2)
    #     test_name = "min_num_rules_2"
    #
    #     result_path = f"..\\cf_results\\cf_metaheuristics\\{test_name}\\{data_name}"
    #     CFEBenchmark.main_benchmark(CFEMetaheuristics, num_classes, non_dominated_solutions, out_path=result_path, test_dataset=test_dataset, data_name=data_name, test_name=test_name)
    #
    # for min_num_rules in [1, 2]:
    #     for data_name in ["bupa", "iris", "pima"]:
    #         num_classes, train_dataset, test_dataset, non_dominated_solutions = get_config(data_name, min_num_rules=min_num_rules)
    #         for param_name in ["sampling_change_fs_params_prob", "sampling_fs_type_prob", "mutation_fs_type_prob", "sampling_noise_str", "mutated_param_prob", "crossover_prob", "mutation_revert_to_initial_prob", "crossover_prob", "mutation_prob"]:
    #             result_path = f"..\\cf_results\\cf_metaheuristics\\param_search\\min_num_rules_{min_num_rules}\\{param_name}\\{data_name}"
    #             CFEBenchmark.param_search(CFEMetaheuristics, num_classes, non_dominated_solutions, result_path, param_name, test_dataset=test_dataset, data_name=data_name)
    #
    #         for param_name in ["n_gen", "pop_size"]:
    #             result_path = f"..\\cf_results\\cf_metaheuristics\\param_search\\min_num_rules_{min_num_rules}\\{param_name}\\{data_name}"
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

    for data_name in ["appendicitis", "bal", "bupa", "contraceptive", "haberman", "heart", "iris", "mammographic", "newthyroid", "page-blocks", "phoneme", "pima", "spectfheart", "tae", "wisconsin", "sonar", "movement_libras", "magic"]:
    # for data_name in ["bupa", "iris", "pima"]:
        all_tests_already_exist = True
        for test_name in test_names:
            test_path = f"..\\cf_results\\cf_metaheuristics\\{test_name}\\{data_name}"
            if not os.path.exists(test_path):
                all_tests_already_exist = False
                break
        if all_tests_already_exist:
            print(f"All tests on {data_name} have already been run, skipping...")
            continue

        num_classes, _, test_dataset, non_dominated_solutions = get_config(data_name)

        for test_name, config in test_configs.items():
            result_path = f"..\\cf_results\\cf_metaheuristics\\{test_name}\\{data_name}"
            CFEBenchmark.main_benchmark(CFEMetaheuristics, num_classes, non_dominated_solutions, out_path=result_path, test_dataset=test_dataset, data_name=data_name, test_name=test_name, **config)

