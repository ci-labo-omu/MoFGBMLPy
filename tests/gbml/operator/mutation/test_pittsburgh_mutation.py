
import copy
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from mofgbmlpy.gbml.operator.crossover.uniform_crossover_single_offspring_michigan import (
    UniformCrossoverSingleOffspringMichigan,
)

from mofgbmlpy.fuzzy.knowledge.factory.homo_triangle_knowledge_factory_2_3_4_5 import (
    HomoTriangleKnowledgeFactory_2_3_4_5,
)

from mofgbmlpy.fuzzy.rule.antecedent.factory.heuristic_antecedent_factory import HeuristicAntecedentFactory

from mofgbmlpy.fuzzy.rule.consequent.learning.learning_basic import LearningBasic

from mofgbmlpy.fuzzy.rule.rule_builder_basic import RuleBuilderBasic

from mofgbmlpy.gbml.solution.michigan_solution_builder import MichiganSolutionBuilder

from mofgbmlpy.gbml.problem.michigan_problem import MichiganProblem
from pymoo.core.population import Population

from mofgbmlpy.gbml.operator.mutation.michigan_mutation import MichiganMutation

from mofgbmlpy.gbml.operator.mutation.pittsburgh_mutation import PittsburghMutation

from mofgbmlpy.fuzzy.classification.single_winner_rule_selection import SingleWinnerRuleSelection

from mofgbmlpy.gbml.problem.pittsburgh_problem import PittsburghProblem

from mofgbmlpy.gbml.objectives.pittsburgh.error_rate import ErrorRate

from mofgbmlpy.gbml.objectives.pittsburgh.num_rules import NumRules
from util import get_a0_0_iris_train_test, create_pittsburgh_sol, create_michigan_sol, \
    helper_init_config, distribution_test_helper_plot_assert
import pytest

def test_java_distribution():
    tests_root = Path(__file__).parents[3]
    tests_data_root = os.path.join(tests_root, "test_data", "mutation", "pittsburgh")

    data_names = [name for name in os.listdir(tests_data_root) if os.path.isdir(os.path.join(tests_data_root, name))]

    for data_name in data_names:
        pop, problem, random_gen = helper_init_config(data_name)
        train = problem.get_training_set()
        knowledge = HomoTriangleKnowledgeFactory_2_3_4_5(train.get_num_dim()).create()

        mutation = PittsburghMutation(knowledge, random_gen)

        sol = pop[0].X[0]
        pop = Population.new(X=np.array([[sol]], dtype=object))

        problem.evaluate(pop.get("X"))

        data_name_config_path = os.path.join(tests_data_root, data_name)

        file_path = os.path.join(data_name_config_path, "offsprings.csv")
        df = pd.read_csv(file_path, header=0)

        file_path = os.path.join(data_name_config_path, "offsprings_rules.csv")
        df_rules = pd.read_csv(file_path, header=0)

        num_iters = len(df)
        error_rate = np.zeros(num_iters)
        num_rules = np.zeros(num_iters)

        rule_weight = []
        rule_length = []
        num_wins = []
        num_classified_patterns = []

        for i in range(num_iters):
            pop = Population.new(X=np.array([[copy.deepcopy(sol)]]))
            new_pop = mutation.do(problem, pop)
            problem.evaluate(new_pop.get("X"))
            child = new_pop[0].X[0]
            error_rate[i] = child.get_error_rate()
            num_rules[i] = child.get_num_vars()

            assert child.get_error_rate() == child.get_objective(0)
            assert child.get_num_vars() == child.get_objective(1)

            for rule in child.get_vars():
                rule_weight.append(rule.get_rule_weight_py().get_value())
                rule_length.append(rule.get_length())
                num_wins.append(rule.get_num_wins())
                num_classified_patterns.append(rule.get_fitness())

        distribution_test_helper_plot_assert(
            error_rate,
            num_rules,
            rule_weight,
            rule_length,
            num_wins,
            num_classified_patterns,
            df,
            df_rules,
            title=f"Comparison on {data_name} using PittsburghMutation on {num_iters} iterations",
        )




