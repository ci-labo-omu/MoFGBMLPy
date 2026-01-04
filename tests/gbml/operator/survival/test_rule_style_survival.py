import copy
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
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

from mofgbmlpy.gbml.operator.selection.nary_tournament_selection_on_fitness import NaryTournamentSelectionOnFitness

from mofgbmlpy.fuzzy.classification.single_winner_rule_selection import SingleWinnerRuleSelection

from mofgbmlpy.gbml.operator.survival.rule_style_survival import RuleStyleSurvival
from util import (
    get_a0_0_iris_train_test,
    create_pittsburgh_sol,
    create_michigan_sol,
    plot_comparison_plot,
    compare_distribution,
)
import pytest


def get_config():
    train, _ = get_a0_0_iris_train_test()

    tests_root = Path(__file__).parents[3]
    tests_data_root = os.path.join(
        tests_root, "test_data", "population_samples", "survival", "rule_addition_style_replacement"
    )

    indices_pop = np.array(json.load(open(os.path.join(tests_data_root, "pop.json"), "r")), int)
    indices_offspring = np.array(json.load(open(os.path.join(tests_data_root, "offspring_pop.json"), "r")), int)

    sols_pop = np.array([create_michigan_sol(train, antecedent_indices=indices) for indices in indices_pop], object)
    create_pittsburgh_sol(train, SingleWinnerRuleSelection(), sols_pop).update_winners_and_errors(train)

    sols_offspring = np.array(
        [create_michigan_sol(train, antecedent_indices=indices) for indices in indices_offspring], object
    )
    create_pittsburgh_sol(train, SingleWinnerRuleSelection(), sols_offspring).update_winners_and_errors(train)

    new_pop_data = json.load(open(os.path.join(tests_data_root, "new_pop.json"), "r"))

    new_pop, new_pop_fitness = np.array(new_pop_data[0], int), np.array(new_pop_data[1], int)

    return sols_pop, sols_offspring, new_pop, new_pop_fitness


def test_valid():
    pop_size = 9
    pop_1_size = int(pop_size * (2 / 3))
    max_num_rules = pop_1_size

    sols_pop, sols_offspring, new_pop_expected_indices, new_pop_fitness_expected = get_config()

    new_pop = RuleStyleSurvival.replace(sols_pop, sols_offspring, max_num_rules=max_num_rules)

    # print("\nPopulation size:", sols_pop.shape[0])
    # for sol in sols_pop:
    #     print(sol)
    #
    # print("\nOffspring population size:", sols_offspring.shape[0])
    # for sol in sols_offspring:
    #     print(sol)
    #
    # print("\nNew population size:", new_pop.shape[0])
    # for sol in new_pop:
    #     print(sol)

    assert new_pop.shape[0] == min(pop_size, max_num_rules)

    assert new_pop.shape[0] == new_pop_expected_indices.shape[0]
    for i in range(new_pop.shape[0]):
        assert np.array_equal(
            np.array(new_pop[i].get_antecedent().get_antecedent_indices(), int), new_pop_expected_indices[i]
        )
        assert new_pop[i].get_fitness() == new_pop_fitness_expected[i]

    #
    # print("pop1:")
    # for i in range(pop1.shape[0]):
    #     print(pop1[i])
    #
    # print("pop2:")
    # for i in range(pop2.shape[0]):
    #     print(pop2[i])
    #
    #
    # print("New population size:", new_pop.shape[0])
    # for i in range(new_pop.shape[0]):
    #     print("Solution", i, "Fitness:", new_pop[i].get_fitness())
    #
    # print("pop1 size:", pop1.shape[0])
    # for i in range(pop1.shape[0]):
    #     print("pop1 Solution", i, "Fitness:", pop1[i].get_fitness())
    #
    # print("pop2 size:", pop2.shape[0])
    # for i in range(pop2.shape[0]):
    #     print("pop2 Solution", i, "Fitness:", pop2[i].get_fitness())

    # import json
    # with open("pop.json", "w") as f:
    #     json.dump([[idx for idx in sol.get_antecedent().get_antecedent_indices()] for sol in pop1], f)

    # with open("offspring_pop.json", "w") as f:
    #     json.dump([[idx for idx in sol.get_antecedent().get_antecedent_indices()] for sol in pop2], f)
