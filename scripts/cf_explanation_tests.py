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
    learner = LearningBasic(runner.get_train_set())

    non_dominated_solutions = res.X
    non_dominated_solutions = remove_duplicates(non_dominated_solutions)

    dataset = learner.get_training_set()


    return dataset, non_dominated_solutions

if __name__ == "__main__":
    # _, non_dominated_solutions = get_config("pima")
    # main_plot_single(non_dominated_solutions)

    # "appendicitis", "bal", "bupa", "contraceptive", "haberman", "heart", "iris", "mammographic", "newthyroid", "page-blocks", "phoneme", "pima", "spectfheart", "tae", "wisconsin", "sonar"

    for data_name in ["appendicitis", "bal", "bupa", "contraceptive", "haberman", "heart", "iris", "mammographic", "newthyroid", "page-blocks", "phoneme", "pima", "spectfheart", "tae", "wisconsin", "sonar"]:
        result_path = f"..\\cf_results\\cf_metaheuristics\\{data_name}"
        dataset, non_dominated_solutions = get_config(data_name)
        main_benchmark(dataset, non_dominated_solutions, out_path=result_path)

    # dataset, non_dominated_solutions = get_config("iris")
    # param_search(dataset, non_dominated_solutions, "..\\cf_results\\cf_metaheuristics_sampling_noise_str_search\\iris", "sampling_noise_str")
