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

from mofgbmlpy.fuzzy.classification.single_winner_rule_selection import SingleWinnerRuleSelection
from util import get_a0_0_iris_train_test, helper_init_config, create_pittsburgh_sol, \
    distribution_test_helper_plot_assert
import pytest


def get_config(prob):
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

    return problem, crossover, pop


@pytest.mark.parametrize("prob", [0, 0.5, 1])
def test_single_offspring_michigan_copy(prob):
    problem, crossover, pop = get_config(prob)
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


def test_uniform_distribution_dataless():
    problem, crossover, pop = get_config(1)

    parents = np.array([[0, 1]])

    num_from_parent_1 = 0
    num_from_parent_2 = 0
    num_iters = 10000
    num_dims = pop[0].X[0].get_num_vars()
    total_known = 0

    for i in range(num_iters):
        offspring = crossover.do(problem, pop, parents=parents)

        # the child must have for each variable either the value from parent 1 or parent 2 (uniform probability)
        for j in range(num_dims):
            is_from_parent_1 = offspring[0].X[0].get_vars()[j] == pop[0].X[0].get_vars()[j]
            is_from_parent_2 = offspring[0].X[0].get_vars()[j] == pop[1].X[0].get_vars()[j]

            assert is_from_parent_1 or is_from_parent_2, f"Variable {j} in offspring is not from either parent"
            if not (is_from_parent_1 and is_from_parent_2):
                # we don't count all variables because if both parents have the same value for a variable we can't
                # determine from which parent it is
                if is_from_parent_1:
                    num_from_parent_1 += 1
                elif is_from_parent_2:
                    num_from_parent_2 += 1
                total_known += 1

    expected_num_from_either_parent = total_known * 0.5

    assert pytest.approx(num_from_parent_1, rel=0.01) == expected_num_from_either_parent
    assert pytest.approx(num_from_parent_2, rel=0.01) == expected_num_from_either_parent


def test_uniform_java_distribution():
    tests_root = Path(__file__).parents[3]
    tests_data_root = os.path.join(tests_root, "test_data", "crossover", "uniform_michigan")

    data_names = [name for name in os.listdir(tests_data_root) if os.path.isdir(os.path.join(tests_data_root, name))]

    for data_name in data_names:
        data_name_config_path = os.path.join(tests_data_root, data_name)

        file_path = os.path.join(data_name_config_path, "offsprings.csv")
        df = pd.read_csv(file_path, header=0)

        file_path = os.path.join(data_name_config_path, "offsprings_rules.csv")
        df_rules = pd.read_csv(file_path, header=0)

        p_pop, p_problem, random_gen = helper_init_config(data_name)
        pop = Population.new(X=np.array([[sol] for sol in p_pop[0].X[0].get_vars()], dtype=object))

        problem, _, _ = get_config(1)

        crossover = UniformCrossoverSingleOffspringMichigan(random_gen, prob=0.9)

        error_rate = []
        num_rules = []

        rule_weight = []
        rule_length = []
        num_wins = []
        num_classified_patterns = []

        num_iters = len(df)

        for _ in range(num_iters):
            p_sol_rules = np.empty(len(pop)-1, dtype=object)
            for i in range(len(pop)-1):
                parents = np.array([[i, i+1]])
                child_m = crossover.do(problem, pop, parents=parents)
                p_sol_rules[i] = child_m[0].X[0]

            p_sol = create_pittsburgh_sol(
                p_problem.get_training_set(), SingleWinnerRuleSelection(), p_sol_rules
            )

            p_problem.evaluate(np.array([[p_sol]], dtype=object))

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
            f"Comparison on {data_name} Using UniformMichiganCrossover on {len(pop)} Michigan solutions on {num_iters} iterations ({len(pop)-1} crossover each)",
        )






