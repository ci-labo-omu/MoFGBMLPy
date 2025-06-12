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
from util import get_a0_0_iris_train_test, create_pittsburgh_sol, create_michigan_sol, plot_comparison_plot, \
    compare_distribution
import pytest

def get_config(pop_size, tournament_size):
    train, _ = get_a0_0_iris_train_test()
    random_gen = np.random.Generator(np.random.MT19937(seed=2022))
    knowledge = HomoTriangleKnowledgeFactory_2_3_4_5(train.get_num_dim()).create()

    selection = NaryTournamentSelectionOnFitness(tournament_size=tournament_size)

    antecedent_factory = HeuristicAntecedentFactory(train, knowledge, False, 0.8, 5, random_gen)
    consequent_factory = LearningBasic(train)
    rule_builder = RuleBuilderBasic(antecedent_factory, consequent_factory, knowledge)

    objectives = np.array([])

    problem = MichiganProblem(objectives, 0, train, rule_builder)

    # michigan_solution_builder = MichiganSolutionBuilder(random_gen, len(objectives), 0, rule_builder)

    tests_root = Path(__file__).parents[3]
    tests_data_root = os.path.join(tests_root, "java_data", "selection", "nary_tournament_selection_on_fitness")
    indices_file = os.path.join(tests_data_root, "michigan_solutions.json")
    indices = np.array(json.load(open(os.path.join(indices_file), "r")), dtype=int)

    sols = []
    for i in range(pop_size):
        sol = [create_michigan_sol(train, antecedent_indices=indices[i])]
        # sol = michigan_solution_builder.create()
        sols.append(sol)

    # import json
    # with open("michigan_solutions.json", "w") as f:
    #     json.dump([[idx for idx in sol[0].get_antecedent().get_antecedent_indices()] for sol in sols], f)

    sols = np.array(sols, dtype=object)
    michigan_solutions = sols.flatten()
    create_pittsburgh_sol(train, SingleWinnerRuleSelection(), michigan_solutions).update_winners_and_errors(train)

    # print("sols")
    # for i, sol in enumerate(sols):
    #     print(f"Solution {i}: {sol[0]}")

    pop = Population.new(X=sols)

    return problem, pop, selection


def test_distribution():
    num_parents = 2
    num_offspring = 100
    pop_size = 100
    tournament_size = 2
    num_iters = 1000

    problem, pop, selection = get_config(pop_size, tournament_size)
    mating_pop = selection.do(problem, pop, num_offspring, n_parents=num_parents, to_pop=False)

    assert mating_pop.shape[0] == num_offspring
    assert mating_pop.shape[1] == num_parents

    selected_indices = np.empty(num_iters * mating_pop.shape[0] * num_parents, dtype=float)
    fitness = np.empty(num_iters * mating_pop.shape[0] * num_parents, dtype=float)

    l = 0
    for i in range(num_iters):
        mating_pop = selection.do(problem, pop, num_offspring, n_parents=num_parents, to_pop=False)
        for j in range(num_offspring):
            for k in range(num_parents):
                fitness[l] = pop[mating_pop[j, k]].X[0].get_fitness()
                selected_indices[l] = mating_pop[j, k]
                l += 1

    tests_root = Path(__file__).parents[3]
    tests_data_root = os.path.join(tests_root, "java_data", "selection", "nary_tournament_selection_on_fitness")
    file_path = os.path.join(tests_data_root, "results.csv")

    java_df = pd.read_csv(file_path, header=0)
    java_fitness = np.array(java_df["num_classified_patterns"].values)
    java_indices = np.array(java_df["rule_index"].values)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs = axs.flatten()
    fig.suptitle(
        f"Comparison of Binary Tournament Selection on Fitness on {num_iters} iterations",
        fontweight="bold",
    )

    max_fitness = max(np.max(fitness), np.max(java_fitness))
    max_indices = max(np.max(selected_indices), np.max(java_indices))

    axs[0] = plot_comparison_plot(axs[0], fitness, java_fitness, "Fitness", x_lim=(0, max_fitness))

    axs[1] = plot_comparison_plot(axs[1], selected_indices, java_indices, "Selected Indices", x_lim=(0, max_indices))

    plt.tight_layout()
    plt.show()

    compare_distribution(java_fitness, fitness, "Fitness")
    compare_distribution(java_indices, selected_indices, "Selected Indices")
