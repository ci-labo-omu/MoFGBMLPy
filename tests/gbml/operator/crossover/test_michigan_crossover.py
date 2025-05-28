import numpy as np
from mofgbmlpy.gbml.operator.crossover.uniform_crossover_single_offspring_michigan import UniformCrossoverSingleOffspringMichigan

from mofgbmlpy.fuzzy.knowledge.factory.homo_triangle_knowledge_factory_2_3_4_5 import \
    HomoTriangleKnowledgeFactory_2_3_4_5

from mofgbmlpy.fuzzy.rule.antecedent.factory.all_combination_antecedent_factory import AllCombinationAntecedentFactory

from mofgbmlpy.fuzzy.rule.consequent.learning.learning_basic import LearningBasic

from mofgbmlpy.fuzzy.rule.rule_builder_basic import RuleBuilderBasic

from mofgbmlpy.gbml.solution.michigan_solution_builder import MichiganSolutionBuilder

from mofgbmlpy.gbml.problem.michigan_problem import MichiganProblem
from pymoo.core.population import Population

from util import get_a0_0_iris_train_test
import pytest


@pytest.mark.parametrize("prob", [0, 0.5, 1])
def test_crossover_copy(prob):
    train, _ = get_a0_0_iris_train_test()
    random_gen = np.random.Generator(np.random.MT19937(seed=2022))
    crossover = UniformCrossoverSingleOffspringMichigan(random_gen, prob=prob)

    knowledge = HomoTriangleKnowledgeFactory_2_3_4_5(train.get_num_dim()).create()
    antecedent_factory = AllCombinationAntecedentFactory(knowledge, random_gen)
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
            assert np.array_equal(offspring[i].X[0].get_vars(), pop[0].X[0].get_vars()) or np.array_equal(offspring[i].X[0].get_vars(), pop[1].X[1].get_vars())
