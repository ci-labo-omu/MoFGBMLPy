import os

from mofgbmlpy.data.input import Input
from mofgbmlpy.explainer.counterfactual_explainer_metaheuristics import main_benchmark, param_search, main_plot_single
from mofgbmlpy.fuzzy.knowledge.factory.homo_triangle_knowledge_factory_2_3_4_5 import \
    HomoTriangleKnowledgeFactory_2_3_4_5

from mofgbmlpy.fuzzy.rule.consequent.learning.learning_basic import LearningBasic
from mofgbmlpy.main.abstract_main import AbstractMain
from mofgbmlpy.main.pittsburgh.pittsburgh_main import PittsburghMain
from mofgbmlpy.explainer.util import remove_duplicates


def get_config(data_name):
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
        "1000",
        "--no-output-files",
        "--objectives",
        "error-rate",
        "num-rules",
    ]

    algo_name = AbstractMain.get_algo_name_from_raw_args(args)
    runner: PittsburghMain = PittsburghMain(HomoTriangleKnowledgeFactory_2_3_4_5, algo_name)
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
    # _, _, _, non_dominated_solutions = get_config("pima")
    # main_plot_single(non_dominated_solutions, use_search_space_crowding=True)

    for data_name in ["appendicitis", "bal", "bupa", "contraceptive", "haberman", "heart", "iris", "mammographic", "newthyroid", "page-blocks", "phoneme", "pima", "spectfheart", "tae", "wisconsin", "sonar"]:
        result_path = f"..\\cf_results\\cf_metaheuristics\\num_features_no_change_loss\\{data_name}"
        try:
            if os.path.exists(result_path):
                print(f"Path {result_path} already exists (skipped).")
                continue
            num_classes, train_dataset, test_dataset, non_dominated_solutions = get_config(data_name)
            main_benchmark(num_classes, non_dominated_solutions, out_path=result_path, objectives=["confidence_loss", "num_changed_features"], test_dataset=test_dataset, data_name=data_name, test_name="num_features_no_change_loss")
        except Exception as e:
            # raise e
            print(f"Error processing {data_name}: {e}")

    for data_name in ["appendicitis", "bal", "bupa", "contraceptive", "haberman", "heart", "iris", "mammographic", "newthyroid", "page-blocks", "phoneme", "pima", "spectfheart", "tae", "wisconsin", "sonar"]:
        result_path = f"..\\cf_results\\cf_metaheuristics\\num_features\\{data_name}"
        try:
            if os.path.exists(result_path):
                print(f"Path {result_path} already exists (skipped).")
                continue
            num_classes, train_dataset, test_dataset, non_dominated_solutions = get_config(data_name)
            main_benchmark(num_classes, non_dominated_solutions, out_path=result_path, objectives=["confidence_loss", "change_loss", "num_changed_features"], test_dataset=test_dataset, data_name=data_name, test_name="num_features")
        except Exception as e:
            print(f"Error processing {data_name}: {e}")

    # "appendicitis", "bal", "bupa", "contraceptive", "haberman", "heart", "iris", "mammographic", "newthyroid", "page-blocks", "phoneme", "pima", "spectfheart", "tae", "wisconsin", "sonar"

    for data_name in ["appendicitis", "bal", "bupa", "contraceptive", "haberman", "heart", "iris", "mammographic", "newthyroid", "page-blocks", "phoneme", "pima", "spectfheart", "tae", "wisconsin", "sonar"]:
        result_path = f"..\\cf_results\\cf_metaheuristics\\classic\\{data_name}"
        try:
            if os.path.exists(result_path):
                print(f"Path {result_path} already exists (skipped).")
                continue
            num_classes, train_dataset, test_dataset, non_dominated_solutions = get_config(data_name)
            main_benchmark(num_classes, non_dominated_solutions, out_path=result_path, test_dataset=test_dataset, data_name=data_name, test_name="classic")
        except Exception as e:
            print(f"Error processing {data_name}: {e}")

    for data_name in ["appendicitis", "bal", "bupa", "contraceptive", "haberman", "heart", "iris", "mammographic", "newthyroid", "page-blocks", "phoneme", "pima", "spectfheart", "tae", "wisconsin", "sonar"]:
        result_path = f"..\\cf_results\\cf_metaheuristics\\X_crowding\\{data_name}"
        try:
            if os.path.exists(result_path):
                print(f"Path {result_path} already exists (skipped).")
                continue
            num_classes, train_dataset, test_dataset, non_dominated_solutions = get_config(data_name)
            main_benchmark(num_classes, non_dominated_solutions, out_path=result_path, use_search_space_crowding=True, test_dataset=test_dataset, data_name=data_name, test_name="X_crowding")
        except Exception as e:
            print(f"Error processing {data_name}: {e}")

    for data_name in ["appendicitis", "bal", "bupa", "contraceptive", "haberman", "heart", "iris", "mammographic", "newthyroid", "page-blocks", "phoneme", "pima", "spectfheart", "tae", "wisconsin", "sonar"]:
        result_path = f"..\\cf_results\\cf_metaheuristics\\error_rate_num_features\\{data_name}"
        try:
            if os.path.exists(result_path):
                print(f"Path {result_path} already exists (skipped).")
                continue
            num_classes, train_dataset, test_dataset, non_dominated_solutions = get_config(data_name)
            main_benchmark(num_classes, non_dominated_solutions, out_path=result_path, objectives=["confidence_loss", "change_loss", "num_changed_features", "train_error_rate"], test_dataset=test_dataset, data_name=data_name, test_name="error_rate_num_features")
        except Exception as e:
            print(f"Error processing {data_name}: {e}")

    for data_name in ["bupa", "iris", "pima"]:
        num_classes, train_dataset, test_dataset, non_dominated_solutions = get_config(data_name)
        for param_name in ["sampling_change_fs_params_prob", "sampling_fs_type_prob", "mutation_fs_type_prob", "sampling_noise_str"]:
            try:
                result_path = f"..\\cf_results\\cf_metaheuristics\\param_search\\{param_name}\\{data_name}"
                if os.path.exists(result_path):
                    print(f"Path {result_path} already exists (skipped).")
                    continue
                param_search(num_classes, non_dominated_solutions, result_path, param_name, test_dataset=test_dataset, data_name=data_name)
            except Exception as e:
                print(f"Error processing {data_name} for param {param_name}: {e}")
