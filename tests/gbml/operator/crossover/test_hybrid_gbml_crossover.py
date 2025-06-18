import json

import numpy as np
from mofgbmlpy.gbml.operator.crossover.uniform_crossover_single_offspring_michigan import (
    UniformCrossoverSingleOffspringMichigan,
)
from scipy.stats import ttest_ind
from mofgbmlpy.fuzzy.knowledge.factory.homo_triangle_knowledge_factory_2_3_4_5 import (
    HomoTriangleKnowledgeFactory_2_3_4_5,
)

from mofgbmlpy.fuzzy.rule.antecedent.factory.heuristic_antecedent_factory import HeuristicAntecedentFactory

from mofgbmlpy.fuzzy.rule.consequent.learning.learning_basic import LearningBasic

from mofgbmlpy.fuzzy.rule.rule_builder_basic import RuleBuilderBasic

from mofgbmlpy.gbml.solution.michigan_solution_builder import MichiganSolutionBuilder
from pathlib import Path
from mofgbmlpy.gbml.problem.michigan_problem import MichiganProblem
from pymoo.core.population import Population

from mofgbmlpy.gbml.solution.pittsburgh_solution import PittsburghSolution

from mofgbmlpy.fuzzy.classification.single_winner_rule_selection import SingleWinnerRuleSelection

from mofgbmlpy.gbml.problem.pittsburgh_problem import PittsburghProblem

from mofgbmlpy.gbml.operator.crossover.pittsburgh_crossover import PittsburghCrossover

from mofgbmlpy.gbml.objectives.pittsburgh.error_rate import ErrorRate

from mofgbmlpy.gbml.objectives.pittsburgh.num_rules import NumRules

from mofgbmlpy.gbml.operator.crossover.hybrid_gbml_crossover import HybridGBMLCrossover

from mofgbmlpy.gbml.operator.crossover.michigan_crossover import MichiganCrossover
from util import (
    get_a0_0_iris_train_test,
    create_pittsburgh_sol,
    create_michigan_sol,
    crossover_test_helper_init_config,
    crossover_test_helper_run,
    get_hybrid_crossover,
)
import pytest
import os


def test_distribution_java():
    tests_root = Path(__file__).parents[3]
    tests_data_root = os.path.join(tests_root, "test_data", "crossover", "hybrid")

    data_names = [name for name in os.listdir(tests_data_root) if os.path.isdir(os.path.join(tests_data_root, name))]

    for data_name in data_names:
        pop, problem, random_gen = crossover_test_helper_init_config(data_name)

        parents = np.array([[0, 1]])

        problem.evaluate(pop.get("X"))

        crossover = get_hybrid_crossover(problem, random_gen)

        data_name_config_path = os.path.join(tests_data_root, data_name)
        crossover_test_helper_run(crossover, problem, pop, parents, data_name, data_name_config_path)
