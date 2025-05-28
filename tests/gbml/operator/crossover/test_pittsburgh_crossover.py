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

from mofgbmlpy.gbml.solution.pittsburgh_solution import PittsburghSolution

from mofgbmlpy.fuzzy.classification.single_winner_rule_selection import SingleWinnerRuleSelection

from mofgbmlpy.gbml.problem.pittsburgh_problem import PittsburghProblem

from mofgbmlpy.gbml.operator.crossover.pittsburgh_crossover import PittsburghCrossover

from mofgbmlpy.gbml.objectives.pittsburgh.error_rate import ErrorRate

from mofgbmlpy.gbml.objectives.pittsburgh.num_rules import NumRules
from util import get_a0_0_iris_train_test, create_pittsburgh_sol, create_michigan_sol
import pytest


@pytest.mark.parametrize("prob", [0, 0.5, 1])
def test_crossover_deepcopy(prob):
    train, _ = get_a0_0_iris_train_test()
    classification = SingleWinnerRuleSelection()

    sol1_num_rules = 2
    sol2_num_rules = 2

    michigan_sols = np.array([create_michigan_sol(train, seed=37+13*i) for i in range(sol1_num_rules)])
    sol1 = create_pittsburgh_sol(train, classification, michigan_sols)
    michigan_sols = np.array([create_michigan_sol(train, seed=37+13*(i+sol1_num_rules)) for i in range(sol2_num_rules)])
    sol2 = create_pittsburgh_sol(train, classification, michigan_sols)

    min_num_rules, max_num_rules = 1, 60

    random_gen = np.random.Generator(np.random.MT19937(seed=2022))
    crossover = PittsburghCrossover(min_num_rules, max_num_rules, random_gen, prob=prob)

    knowledge = HomoTriangleKnowledgeFactory_2_3_4_5(train.get_num_dim()).create()
    antecedent_factory = AllCombinationAntecedentFactory(knowledge, random_gen)
    consequent_factory = LearningBasic(train)
    rule_builder = RuleBuilderBasic(antecedent_factory, consequent_factory, knowledge)

    objectives = np.array([])
    michigan_solution_builder = MichiganSolutionBuilder(random_gen, len(objectives), 0, rule_builder)

    num_vars = train.get_num_dim()
    objectives = np.array([ErrorRate(train), NumRules()])

    problem = PittsburghProblem(num_vars, objectives, 0, train, michigan_solution_builder, classification)

    pop = Population.new(X=np.array([[sol1], [sol2]], dtype=object))
    parents = np.array([[0, 1], [1, 0]])

    offspring = crossover.do(problem, pop, parents=parents)

    print(f"\n(Prob = {prob})\nParents:")
    print(pop[0].X)
    print(pop[1].X)
    print("Offspring:")
    print(offspring[0].X)
    print(offspring[1].X)

    assert offspring.shape == (2,)
    assert id(offspring[0].X) != id(offspring[1].X)

    for i in range(2):
        for j in range(2):
            assert id(offspring[i].X[0]) != id(pop[j].X[0])
            assert id(offspring[i].X[0].get_vars().base) != id(pop[j].X[0].get_vars().base)
            for ki in range(len(offspring[i].X[0].get_vars())):
                for kj in range(len(pop[j].X[0].get_vars())):
                    assert id(offspring[i].X[0].get_vars()[ki]) != id(pop[j].X[0].get_vars()[kj])
                    # check michigan solutions antecedents
                    assert id(offspring[i].X[0].get_vars()[ki].get_antecedent().get_antecedent_indices().base) != id(pop[j].X[0].get_vars()[kj].get_antecedent().get_antecedent_indices().base)

    if prob == 0:
        for i in range(2):
            lo = len(offspring[i].X[0].get_vars())
            lp1 = len(pop[0].X[0].get_vars())
            lp2 = len(pop[1].X[0].get_vars())
            assert lo == lp1 or lo == lp2

            off_vars = offspring[i].X[0].get_vars()
            if lo == lp1 and lo == lp2:
                pop_vars1 = pop[0].X[0].get_vars()
                pop_vars2 = pop[1].X[0].get_vars()
                is_different_pop_1 = False
                is_different_pop_2 = False

                for j in range(len(off_vars)):
                    if off_vars[j] != pop_vars1[j]:
                        is_different_pop_1 = True
                    if off_vars[j] != pop_vars2[j]:
                        is_different_pop_2 = True
                assert (is_different_pop_1 and not is_different_pop_2) or (is_different_pop_2 and not is_different_pop_1), "Offspring is not a copy of at one parent"

            elif lo == lp1:
                pop_vars = pop[0].X[0].get_vars()

                for j in range(len(off_vars)):
                    assert off_vars[j] == pop_vars[j]
            elif lo == lp2:
                pop_vars = pop[1].X[0].get_vars()

                for j in range(len(off_vars)):
                    assert off_vars[j] == pop_vars[j]
