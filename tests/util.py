import csv
import math
import os

import numpy as np

from mofgbmlpy.data.input import Input
from mofgbmlpy.fuzzy.knowledge.factory.homo_triangle_knowledge_factory_2_3_4_5 import \
    HomoTriangleKnowledgeFactory_2_3_4_5
from mofgbmlpy.fuzzy.rule.antecedent.factory.all_combination_antecedent_factory import AllCombinationAntecedentFactory
from mofgbmlpy.fuzzy.rule.antecedent.factory.heuristic_antecedent_factory import HeuristicAntecedentFactory
from mofgbmlpy.fuzzy.rule.consequent.learning.learning_basic import LearningBasic
from mofgbmlpy.fuzzy.rule.consequent.learning.learning_multi import LearningMulti
from mofgbmlpy.fuzzy.rule.rule_builder_basic import RuleBuilderBasic
from mofgbmlpy.fuzzy.rule.rule_builder_multi import RuleBuilderMulti
from mofgbmlpy.gbml.solution.michigan_solution import MichiganSolution
from mofgbmlpy.main.arguments.arguments import Arguments


def get_datasets(datasets_dir="../dataset"):
    datasets = {}
    for folder in os.listdir(datasets_dir):
        datasets[folder] = []
        for items in os.walk(os.path.join(datasets_dir, folder)):
            if "subdata" in items[0].split(os.sep):
                continue
            files = []
            for file in items[2]:
                path = os.path.join(items[0], file)
                try:
                    with open(path, newline='') as f:
                        reader = csv.reader(f)
                        header = next(reader)
                        if len(header) < 3:
                            raise Exception("Invalid header")
                        for i in range(3):
                            int(header[i])
                    files.append(path)
                except:
                    pass  # Invalid format (not csv or invalid header)

            datasets[folder] += files
    return datasets


def get_a0_0_iris_train_test():
    root_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    args = Arguments()
    args.set("TRAIN_FILE", f"{root_folder}/dataset/iris/a0_0_iris-10tra.dat")
    args.set("TEST_FILE", f"{root_folder}/dataset/iris/a0_0_iris-10tst.dat")
    args.set("IS_MULTI_LABEL", False)

    return Input.get_train_test_files(args)


def get_a0_0_german_train_test():
    root_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    args = Arguments()
    args.set("TRAIN_FILE", f"{root_folder}/dataset/german/a0_0_german-10tra.dat")
    args.set("TEST_FILE", f"{root_folder}/dataset/german/a0_0_german-10tst.dat")
    args.set("IS_MULTI_LABEL", True)

    return Input.get_train_test_files(args)


def create_michigan_sol(training_data_set, seed=2022, antecedent_indices=None, consequent=None, is_multi_label=False):
    random_gen = np.random.Generator(np.random.MT19937(seed))

    knowledge = HomoTriangleKnowledgeFactory_2_3_4_5(training_data_set.get_num_dim()).create()
    antecedent_factory = HeuristicAntecedentFactory(training_data_set,
                                                    knowledge,
                                                    False,
                                                    0.7,
                                                    5,
                                                    random_gen)

    if is_multi_label:
        consequent_factory = LearningMulti(training_data_set)
        rule_builder = RuleBuilderMulti(antecedent_factory, consequent_factory, knowledge)
    else:
        consequent_factory = LearningBasic(training_data_set)
        rule_builder = RuleBuilderBasic(antecedent_factory, consequent_factory, knowledge)


    solution = MichiganSolution(random_gen,
                                2,
                                0,
                                rule_builder)



    if antecedent_indices is not None:
        solution.set_vars(antecedent_indices)
        solution.learning()

    if consequent is not None:
        solution.get_rule().set_consequent(consequent)

    return solution


def float_eq(value1, value2, precision=1e-6):
    return abs(value1 - value2) < precision
