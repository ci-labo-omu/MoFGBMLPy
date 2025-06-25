import copy
import numpy as np
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
from util import get_a0_0_iris_train_test
import pytest


def get_config(mutation_rt):
    train, _ = get_a0_0_iris_train_test()
    random_gen = np.random.Generator(np.random.MT19937(seed=2022))
    knowledge = HomoTriangleKnowledgeFactory_2_3_4_5(train.get_num_dim()).create()

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
            np.isclose(probabilities, expected_prob, atol=0.1)
        ), f"Distribution of values for dimension {j} is not uniform: {probabilities}"


def test_mutation_rt_0():
    mutation_rt = 0
    problem, sol, sol_vars, num_dims, num_iters, mutation = get_config(mutation_rt)

    for i in range(num_iters):
        pop = Population.new(X=np.array([[copy.deepcopy(sol)]]))
        new_sol = mutation.do(problem, pop)[0].X[0]
        new_sol_vars = new_sol.get_vars()
        assert np.array_equal(new_sol_vars, sol_vars), f"Mutation did change the solution"
