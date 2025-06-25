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

from mofgbmlpy.gbml.operator.crossover.michigan_crossover import MichiganCrossover

from mofgbmlpy.gbml.operator.mutation.michigan_mutation import MichiganMutation

from mofgbmlpy.gbml.operator.selection.nary_tournament_selection_on_fitness import NaryTournamentSelectionOnFitness

from mofgbmlpy.fuzzy.classification.single_winner_rule_selection import SingleWinnerRuleSelection
from util import (
    get_a0_0_iris_train_test,
    crossover_test_helper_init_config,
    crossover_test_helper_run,
    plot_comparison_plot,
    compare_distribution,
    create_pittsburgh_sol,
    crossover_test_helper_plot_assert,
)
import pytest


@pytest.mark.parametrize("prob", [0, 0.5, 1])
def test_crossover_copy(prob):
    train, _ = get_a0_0_iris_train_test()
    random_gen = np.random.Generator(np.random.MT19937(seed=2022))
    crossover = UniformCrossoverSingleOffspringMichigan(random_gen, prob=prob)

    knowledge = HomoTriangleKnowledgeFactory_2_3_4_5(train.get_num_dim()).create()
    antecedent_factory = HeuristicAntecedentFactory(train, knowledge, False, 0.8, 5, random_gen)
    consequent_factory = LearningBasic(train)
    rule_builder = RuleBuilderBasic(antecedent_factory, consequent_factory, knowledge)

    objectives = np.array([])
    michigan_solution_builder = MichiganSolutionBuilder(random_gen, len(objectives), 0, rule_builder)

    sol1 = michigan_solution_builder.create()
    sol2 = michigan_solution_builder.create()

    problem = MichiganProblem(objectives, 0, train, rule_builder)

    pop = Population.new(X=np.array([sol1, sol2]))
    parents = np.array([[0, 1], [1, 0]])

    offspring = crossover.do(problem, pop, parents=parents)

    # print("\nParents:")
    # print(pop[0].X)
    # print(pop[1].X)
    # print("Offspring:")
    # print(offspring[0].X)
    # print(offspring[1].X)

    assert offspring.shape == (2,)
    assert id(offspring[0].X) != id(offspring[1].X)

    for i in range(2):
        for j in range(2):
            assert id(offspring[i].X[0]) != id(pop[j].X[0])
            assert id(offspring[i].X[0].get_vars().base) != id(pop[j].X[0].get_vars().base)

    if prob == 0:
        for i in range(2):
            assert np.array_equal(offspring[i].X[0].get_vars(), pop[0].X[0].get_vars()) or np.array_equal(
                offspring[i].X[0].get_vars(), pop[1].X[0].get_vars()
            )


def test_distribution_java():
    max_num_rules = 60
    michigan_crossover_probability = 1.0
    rule_change_rate = 0.2

    tests_root = Path(__file__).parents[3]
    tests_data_root = os.path.join(tests_root, "test_data", "crossover", "michigan")

    data_names = [name for name in os.listdir(tests_data_root) if os.path.isdir(os.path.join(tests_data_root, name))]

    for data_name in data_names:
        pop, problem, random_gen = crossover_test_helper_init_config(data_name)
        parents = np.array([[0]])

        problem.evaluate(pop.get("X"))

        crossover = MichiganCrossover(
            rule_change_rate,
            problem.get_training_set(),
            problem.get_knowledge(),
            max_num_rules,
            random_gen,
            michigan_crossover_probability,
        )

        data_name_config_path = os.path.join(tests_data_root, data_name)
        crossover_test_helper_run(crossover, problem, pop, parents, data_name, data_name_config_path)


def test_distribution_ga_rules_gen():
    # set seed of pymoo
    np.random.seed(2022)

    max_num_rules = 60
    michigan_crossover_probability = 0.9
    rule_change_rate = 0.2

    tests_root = Path(__file__).parents[3]
    tests_data_root = os.path.join(tests_root, "test_data", "crossover", "michigan")

    data_names = [name for name in os.listdir(tests_data_root) if os.path.isdir(os.path.join(tests_data_root, name))]

    for data_name in data_names:
        pop, problem, random_gen = crossover_test_helper_init_config(data_name)

        problem.evaluate(pop.get("X"))

        crossover = MichiganCrossover(
            rule_change_rate,
            problem.get_training_set(),
            problem.get_knowledge(),
            max_num_rules,
            random_gen,
            michigan_crossover_probability,
        )

        michigan_problem = MichiganProblem(
            [],  # Objectives are not used
            problem.get_num_constraints(),
            problem.get_training_set(),
            problem.get_rule_builder(),
        )

        m_crossover = UniformCrossoverSingleOffspringMichigan(random_gen, michigan_crossover_probability)

        mutation_rt = 1 / problem.get_training_set().get_num_dim()
        mutation = MichiganMutation(problem.get_knowledge(), mutation_rt, random_gen)

        parent = pop[0].X[0]

        if parent.get_num_vars() == 1:
            # no crossover
            tournament_size = 1
        else:
            tournament_size = 2
        selection = NaryTournamentSelectionOnFitness(tournament_size)

        data_name_config_path = os.path.join(tests_data_root, data_name)

        ga_gen_files_path = os.path.join(data_name_config_path, "ga_gen")
        file_path = os.path.join(ga_gen_files_path, "offsprings_rules.csv")
        df_rules = pd.read_csv(file_path, header=0)

        file_path = os.path.join(ga_gen_files_path, "offsprings.csv")
        df = pd.read_csv(file_path, header=0)

        num_ga = len(df_rules) // len(df)
        num_offspring = len(df)

        error_rate = []
        num_rules = []

        rule_length = []
        num_wins = []
        num_classified_patterns = []
        rule_weight = []

        for i in range(num_offspring):
            michigan_solutions_array = np.empty((parent.get_num_vars(), 1), dtype=object)
            parent_vars = parent.get_vars()
            for j in range(michigan_solutions_array.shape[0]):
                michigan_solutions_array[j, 0] = parent_vars[j]
            michigan_population = Population.new(X=michigan_solutions_array)

            generated_solutions = crossover.ga_rules_gen(
                m_crossover, mutation, selection, michigan_population, michigan_problem, num_ga, 2
            )
            for sol in generated_solutions:
                for var in parent_vars:
                    assert id(var) != id(sol)
                    assert id(var.get_vars().base) != id(sol.get_vars().base)

            p_sol = create_pittsburgh_sol(
                problem.get_training_set(), SingleWinnerRuleSelection(), np.array(generated_solutions, object)
            )

            p_sol.update_winners_and_errors(problem.get_training_set())
            problem.evaluate(np.array([[p_sol]], dtype=object))

            error_rate.append(p_sol.get_error_rate())
            num_rules.append(p_sol.get_num_vars())

            assert p_sol.get_error_rate() == p_sol.get_objective(0)
            assert p_sol.get_num_vars() == p_sol.get_objective(1)

            for sol in generated_solutions:
                rule_length.append(sol.get_rule().get_length())
                num_wins.append(sol.get_num_wins())
                num_classified_patterns.append(sol.get_fitness())
                rule_weight.append(sol.get_rule_weight_py().get_value())

        error_rate = np.array(error_rate)
        num_rules = np.array(num_rules)
        rule_length = np.array(rule_length)
        num_wins = np.array(num_wins)
        num_classified_patterns = np.array(num_classified_patterns)
        rule_weight = np.array(rule_weight)

        crossover_test_helper_plot_assert(
            error_rate,
            num_rules,
            rule_weight,
            rule_length,
            num_wins,
            num_classified_patterns,
            df,
            df_rules,
            f"Comparison on {data_name} Using GA Rules Generation on {num_offspring} solutions with {num_ga} rules",
        )


def test_distribution_heuristic_rules_gen():
    max_num_rules = 60
    michigan_crossover_probability = 1.0
    rule_change_rate = 0.2

    tests_root = Path(__file__).parents[3]
    tests_data_root = os.path.join(tests_root, "test_data", "crossover", "michigan")

    data_names = [name for name in os.listdir(tests_data_root) if os.path.isdir(os.path.join(tests_data_root, name))]

    for data_name in data_names:
        pop, problem, random_gen = crossover_test_helper_init_config(data_name)

        problem.evaluate(pop.get("X"))

        crossover = MichiganCrossover(
            rule_change_rate,
            problem.get_training_set(),
            problem.get_knowledge(),
            max_num_rules,
            random_gen,
            michigan_crossover_probability,
        )

        parent = pop[0].X[0]

        data_name_config_path = os.path.join(tests_data_root, data_name)
        ga_gen_files_path = os.path.join(data_name_config_path, "heuristic_gen")
        file_path = os.path.join(ga_gen_files_path, "offsprings_rules.csv")
        df_rules = pd.read_csv(file_path, header=0)

        file_path = os.path.join(ga_gen_files_path, "offsprings.csv")
        df = pd.read_csv(file_path, header=0)

        num_heuristic = len(df_rules) // len(df)
        num_offspring = len(df)

        error_rate = []
        num_rules = []

        rule_length = []
        num_wins = []
        num_classified_patterns = []
        rule_weight = []

        for i in range(num_offspring):
            michigan_solutions_array = np.empty((parent.get_num_vars(), 1), dtype=object)
            parent_vars = parent.get_vars()
            for j in range(michigan_solutions_array.shape[0]):
                michigan_solutions_array[j, 0] = parent_vars[j]

            generated_solutions = crossover.heuristic_rules_gen(parent, num_heuristic)

            p_sol = create_pittsburgh_sol(
                problem.get_training_set(), SingleWinnerRuleSelection(), np.array(generated_solutions, object)
            )

            p_sol.update_winners_and_errors(problem.get_training_set())
            problem.evaluate(np.array([[p_sol]], dtype=object))

            error_rate.append(p_sol.get_error_rate())
            num_rules.append(p_sol.get_num_vars())

            assert p_sol.get_error_rate() == p_sol.get_objective(0)
            assert p_sol.get_num_vars() == p_sol.get_objective(1)

            for sol in generated_solutions:
                rule_length.append(sol.get_rule().get_length())
                num_wins.append(sol.get_num_wins())
                num_classified_patterns.append(sol.get_fitness())
                rule_weight.append(sol.get_rule_weight_py().get_value())

        error_rate = np.array(error_rate)
        num_rules = np.array(num_rules)
        rule_length = np.array(rule_length)
        num_wins = np.array(num_wins)
        num_classified_patterns = np.array(num_classified_patterns)
        rule_weight = np.array(rule_weight)

        crossover_test_helper_plot_assert(
            error_rate,
            num_rules,
            rule_weight,
            rule_length,
            num_wins,
            num_classified_patterns,
            df,
            df_rules,
            f"Comparison on {data_name} Using Heuristic Rules Generation on {num_offspring} solutions with {num_heuristic} rules",
        )
