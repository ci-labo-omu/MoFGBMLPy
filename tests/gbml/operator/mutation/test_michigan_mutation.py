import copy
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

from mofgbmlpy.fuzzy.classification.single_winner_rule_selection import SingleWinnerRuleSelection
from util import get_a0_0_iris_train_test, helper_init_config, create_pittsburgh_sol, \
    distribution_test_helper_plot_assert
import pytest


def get_config(mutation_rt=None):
    train, _ = get_a0_0_iris_train_test()
    random_gen = np.random.Generator(np.random.MT19937(seed=2022))
    knowledge = HomoTriangleKnowledgeFactory_2_3_4_5(train.get_num_dim()).create()

    if mutation_rt is None:
        mutation_rt = 1 / train.get_num_dim()

    mutation = MichiganMutation(knowledge, mutation_rt, random_gen)
    antecedent_factory = HeuristicAntecedentFactory(train, knowledge, False, 0.8, 5, random_gen)
    consequent_factory = LearningBasic(train)
    rule_builder = RuleBuilderBasic(antecedent_factory, consequent_factory, knowledge)

    objectives = np.array([])
    michigan_solution_builder = MichiganSolutionBuilder(random_gen, len(objectives), 0, rule_builder)

    sol = michigan_solution_builder.create()[0]
    sol_vars = sol.get_vars()

    problem = MichiganProblem(objectives, 0, train, rule_builder)

    num_dims = train.get_num_dim()

    num_iters = 10000

    return problem, sol, sol_vars, num_dims, num_iters, mutation


def test_mutation_rt_1():
    mutation_rt = 1
    problem, sol, sol_vars, num_dims, num_iters, mutation = get_config(mutation_rt)

    offsprings = np.empty(num_iters, dtype=object)

    for i in range(num_iters):
        pop = Population.new(X=np.array([[copy.deepcopy(sol)]]))
        new_sol = mutation.do(problem, pop)[0].X[0]
        new_sol_vars = new_sol.get_vars()
        assert not np.array_equal(new_sol_vars, sol_vars), f"Mutation did not change the solution"
        offsprings[i] = new_sol

    # uniform distribution check
    for j in range(num_dims):
        values = [offspring.get_vars()[j] for offspring in offsprings]
        unique_values, counts = np.unique(values, return_counts=True)

        probabilities = counts / num_iters
        expected_prob = 1 / len(unique_values)

        # Check if the distribution is uniform
        assert np.all(
            np.isclose(probabilities, expected_prob, atol=0.01)
        ), f"Distribution of values for dimension {j} is not uniform: {probabilities}"


def test_mutation_rt_0():
    mutation_rt = 0
    problem, sol, sol_vars, num_dims, num_iters, mutation = get_config(mutation_rt)

    for i in range(num_iters):
        pop = Population.new(X=np.array([[copy.deepcopy(sol)]]))
        new_sol = mutation.do(problem, pop)[0].X[0]
        new_sol_vars = new_sol.get_vars()
        assert np.array_equal(new_sol_vars, sol_vars), f"Mutation did change the solution"


def test_distribution_java():
    tests_root = Path(__file__).parents[3]
    tests_data_root = os.path.join(tests_root, "test_data", "mutation", "michigan")

    data_names = [name for name in os.listdir(tests_data_root) if os.path.isdir(os.path.join(tests_data_root, name))]

    for data_name in data_names:
        data_name_config_path = os.path.join(tests_data_root, data_name)

        file_path = os.path.join(data_name_config_path, "offsprings.csv")
        df = pd.read_csv(file_path, header=0)

        file_path = os.path.join(data_name_config_path, "offsprings_rules.csv")
        df_rules = pd.read_csv(file_path, header=0)

        pop, problem, random_gen = helper_init_config(data_name)
        mutation_rt = 1 / problem.get_training_set().get_num_dim()
        mutation = MichiganMutation(problem.get_knowledge(), mutation_rt, random_gen)

        pop = Population.new(X=np.array([[v] for v in pop[0].X[0].get_vars()], object))

        error_rate = []
        num_rules = []

        rule_weight = []
        rule_length = []
        num_wins = []
        num_classified_patterns = []

        num_iters = len(df)

        for _ in range(num_iters):
            pop_copy = Population.new(X=np.array([[copy.deepcopy(ind[0])] for ind in pop.get("X")], object))
            new_pop = mutation.do(problem, pop_copy)
            rules = new_pop.get("X")[:,0]

            p_sol = create_pittsburgh_sol(
                problem.get_training_set(), SingleWinnerRuleSelection(), np.array(rules, object)
            )

            p_sol.learning()

            indices_to_remove = []
            for i in range(len(p_sol.get_vars())):
                if p_sol.get_var(i).get_rule().is_rejected_class_label():
                    indices_to_remove.append(i)
            p_sol.remove_vars(np.array(indices_to_remove, int))

            problem.evaluate(np.array([[p_sol]], object))

            error_rate.append(p_sol.get_error_rate())
            num_rules.append(p_sol.get_num_vars())
            assert p_sol.get_error_rate() == p_sol.get_objective(0)
            assert p_sol.get_num_vars() == p_sol.get_objective(1)

            for rule in p_sol.get_vars():
                rule_length.append(rule.get_rule().get_length())
                num_wins.append(rule.get_num_wins())
                num_classified_patterns.append(rule.get_fitness())
                rule_weight.append(rule.get_rule_weight_py().get_value())

        distribution_test_helper_plot_assert(
            error_rate,
            num_rules,
            rule_weight,
            rule_length,
            num_wins,
            num_classified_patterns,
            df,
            df_rules,
            f"Comparison on {data_name} Using Michigan Mutation on {len(pop)} Michigan solutions on {num_iters} iterations",
        )
