import os

from mofgbmlpy.data.input import Input
from mofgbmlpy.fuzzy.knowledge.factory.homo_triangle_knowledge_factory_2_3_4_5 import \
    HomoTriangleKnowledgeFactory_2_3_4_5

from mofgbmlpy.fuzzy.rule.consequent.learning.learning_basic import LearningBasic
from mofgbmlpy.main.abstract_main import AbstractMain
from mofgbmlpy.main.pittsburgh.pittsburgh_main import PittsburghMain
from mofgbmlpy.explainer.util import remove_duplicates
from mofgbmlpy.explainer.counterfactual_explainer_benchmark import CounterFactualExplainerBenchmark as CFEBenchmark
from mofgbmlpy.explainer.counterfactual_explainer_gradient import CounterFactualExplainerGradient as CFEGradient


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

    return num_classes, train_set, test_set, non_dominated_solutions

if __name__ == "__main__":
    # _, _, test_dataset, non_dominated_solutions = get_config("pima")
    # CFEBenchmark.main_plot_single(CFEGradient, non_dominated_solutions, sol_index=2, test_dataset=test_dataset)

    #
    # for data_name in ["contraceptive", "appendicitis", "bal", "bupa", "haberman", "heart", "iris", "mammographic", "newthyroid", "page-blocks", "phoneme", "pima", "spectfheart", "tae", "wisconsin", "sonar"]:
    #     num_classes, _, test_dataset, non_dominated_solutions = get_config(data_name, min_num_rules=2)
    #
    #     result_path = f"..\\cf_results\\cf_gradient\\min_num_rules_2\\{data_name}"
    #     CFEBenchmark.main_benchmark(CFEGradient, num_classes, non_dominated_solutions, out_path=result_path, test_dataset=test_dataset, data_name=data_name, test_name="min_num_rules_2")
    #
    #
    # for min_num_rules in [1]: #[1, 2]:
    #     for data_name in ["appendicitis", "bupa", "iris", "pima"]:
    #         num_classes, train_dataset, test_dataset, non_dominated_solutions = get_config(data_name, min_num_rules=min_num_rules)
    #
    #         result_path = f"..\\cf_results\\cf_gradient\\param_search\\min_num_rules_{min_num_rules}\\confidence_loss_weight\\{data_name}"
    #         CFEBenchmark.param_search(CFEGradient, num_classes, non_dominated_solutions, result_path, "confidence_loss_weight", test_dataset=test_dataset, data_name=data_name)
    #
    #         result_path = f"..\\cf_results\\cf_gradient\\param_search\\min_num_rules_{min_num_rules}\\learning_rate\\{data_name}"
    #         CFEBenchmark.param_search(CFEGradient, num_classes, non_dominated_solutions, result_path, "learning_rate", test_dataset=test_dataset, data_name=data_name, min_val=0.1, max_val=5.0, num_experiments=50)
    #
    #         result_path = f"..\\cf_results\\cf_gradient\\param_search\\min_num_rules_{min_num_rules}\\max_num_epochs\\{data_name}"
    #         CFEBenchmark.param_search(CFEGradient, num_classes, non_dominated_solutions, result_path, "max_num_epochs", test_dataset=test_dataset, data_name=data_name, min_val=10, max_val=100, num_experiments=10, is_int=True)

    # for data_name in ["appendicitis", "bal", "bupa", "contraceptive", "haberman", "heart", "iris", "mammographic", "newthyroid", "page-blocks", "phoneme", "pima", "spectfheart", "tae", "wisconsin", "sonar", "magic", "movement_libras"]:
    for data_name in ["bupa", "iris", "pima"]:
    # for data_name in ["movement_libras"]:
        num_classes, _, test_dataset, non_dominated_solutions = get_config(data_name)

        result_path = f"..\\cf_results\\cf_gradient\\classic\\{data_name}"
        CFEBenchmark.main_benchmark(CFEGradient, num_classes, non_dominated_solutions, out_path=result_path, test_dataset=test_dataset, data_name=data_name, test_name="classic")
