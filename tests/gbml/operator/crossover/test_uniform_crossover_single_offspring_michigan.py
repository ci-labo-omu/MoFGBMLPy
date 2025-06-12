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

from util import get_a0_0_iris_train_test
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

def test_uniform_distribution():
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
