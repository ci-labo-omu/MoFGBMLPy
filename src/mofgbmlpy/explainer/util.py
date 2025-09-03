import numpy as np
from mofgbmlpy.fuzzy.rule.consequent.learning.learning_basic import LearningBasic
from mofgbmlpy.fuzzy.knowledge.factory.homo_triangle_knowledge_factory_2_3_4_5 import (
    HomoTriangleKnowledgeFactory_2_3_4_5,
)
from pymoo.core.population import Population

from mofgbmlpy.main.abstract_main import AbstractMain
from mofgbmlpy.main.pittsburgh.pittsburgh_main import PittsburghMain


def exists_in_sols(sol, sols_list):
    for filtered_sol in sols_list:
        if sol[0] == filtered_sol[0]:
            return True
    return False


def remove_duplicates(sols):
    filtered_sols = []
    for sol in sols:
        if not exists_in_sols(sol, filtered_sols):
            filtered_sols.append(sol)
    return np.array(filtered_sols, dtype=object)


def get_config(data_name):
    args = [
        "--data-name",
        f"{data_name}",
        "--algorithm-id",
        "0",
        "--experiment-id",
        "0",
        "--train-file",
        f"..\\..\\..\\dataset\\{data_name}\\a0_0_{data_name}-10tra.dat",
        "--test-file",
        f"..\\..\\..\\dataset\\{data_name}\\a0_0_{data_name}-10tra.dat",
        "--terminate-evaluation",
        "1000",
        "--no-output-files",
        "--objectives",
        "error-rate",
        "num-rules",
    ]

    algo_name = AbstractMain.get_algo_name_from_raw_args(args)
    runner = PittsburghMain(HomoTriangleKnowledgeFactory_2_3_4_5, algo_name)
    res = runner.run(args)
    learner = LearningBasic(runner.get_train_set())

    non_dominated_solutions = res.X
    non_dominated_solutions = remove_duplicates(non_dominated_solutions)

    dataset = learner.get_training_set()


    return dataset, non_dominated_solutions, learner
