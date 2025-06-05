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


def get_config(sol1_num_rules, sol2_num_rules):
    train, _ = get_a0_0_iris_train_test()
    classification = SingleWinnerRuleSelection()

    michigan_sols = np.array([create_michigan_sol(train, seed=37 + 13 * i) for i in range(sol1_num_rules)])
    sol1 = create_pittsburgh_sol(train, classification, michigan_sols)
    michigan_sols = np.array(
        [create_michigan_sol(train, seed=37 + 13 * (i + sol1_num_rules)) for i in range(sol2_num_rules)])
    sol2 = create_pittsburgh_sol(train, classification, michigan_sols)

    random_gen = np.random.Generator(np.random.MT19937(seed=2022))

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

    return pop, problem, parents


@pytest.mark.parametrize("prob", [0, 0.5, 1])
@pytest.mark.parametrize("sol1_num_rules", [1, 2, 3])
@pytest.mark.parametrize("sol2_num_rules", [1, 2, 3])
def test_crossover_deepcopy(prob, sol1_num_rules, sol2_num_rules):
    min_num_rules, max_num_rules = 1, 60

    random_gen = np.random.Generator(np.random.MT19937(seed=2022))
    pop, problem, parents = get_config(sol1_num_rules, sol2_num_rules)

    crossover = PittsburghCrossover(min_num_rules, max_num_rules, random_gen, prob=prob)
    offspring = crossover.do(problem, pop, parents=parents)

    # print(f"\n(Prob = {prob})\nParents:")
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
            for ki in range(len(offspring[i].X[0].get_vars())):
                for kj in range(len(pop[j].X[0].get_vars())):
                    assert id(offspring[i].X[0].get_vars()[ki]) != id(pop[j].X[0].get_vars()[kj])
                    # check michigan solutions antecedents
                    assert id(offspring[i].X[0].get_vars()[ki].get_antecedent().get_antecedent_indices().base) != id(pop[j].X[0].get_vars()[kj].get_antecedent().get_antecedent_indices().base)




@pytest.mark.parametrize("prob", [0, 0.5, 1])
@pytest.mark.parametrize("sol1_num_rules", [1, 2, 3])
@pytest.mark.parametrize("sol2_num_rules", [1, 2, 3])
def test_crossover_output(prob, sol1_num_rules, sol2_num_rules):
    min_num_rules, max_num_rules = 1, 60
    random_gen = np.random.Generator(np.random.MT19937(seed=2022))
    pop, problem, parents = get_config(sol1_num_rules, sol2_num_rules)
    crossover = PittsburghCrossover(min_num_rules, max_num_rules, random_gen, prob=prob)
    offspring = crossover.do(problem, pop, parents=parents)

    # print(f"\n(Prob = {prob})\nParents:")
    # print(pop[0].X)
    # print(pop[1].X)
    # print("Offspring:")
    # print(offspring[0].X)
    # print(offspring[1].X)

    assert offspring.shape == (2,)

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
    else:
        # the offspring should have some rules from parent 1 (unique) and some from parent 2 (unique)
        for i in range(2):
            off_vars = offspring[i].X[0].get_vars()
            p1_vars = pop[0].X[0].get_vars()
            p2_vars = pop[1].X[0].get_vars()

            for j in range(len(off_vars)):
                is_from_parent = False
                for k in range(len(p1_vars)):
                    if off_vars[j] == p1_vars[k]:
                        # the rule is from parent 1
                        is_from_parent = True
                        break

                for k in range(len(p2_vars)):
                    if off_vars[j] == p2_vars[k]:
                        # the rule is from parent 2
                        is_from_parent = True
                        break

                assert is_from_parent, "Offspring rule is not from any parent"

@pytest.mark.parametrize("min_num_rules, max_num_rules", [(1, 60), (5, 30), (20, 40)])
@pytest.mark.parametrize("num_rules_p1, num_rules_p2", [(25, 25), (10, 50), (50, 10), (40, 40)])
def test_get_num_rules_from_parents_uniform(min_num_rules, max_num_rules, num_rules_p1, num_rules_p2):
    min_num_rules, max_num_rules = 1, 60
    random_gen = np.random.Generator(np.random.MT19937(seed=2022))
    crossover = PittsburghCrossover(min_num_rules, max_num_rules, random_gen)

    num_rules_from_p1, num_rules_from_p2 = [], []

    try:
        for i in range(10000):
            from_p1, from_p2 = crossover.get_num_rules_from_parents(num_rules_p1, num_rules_p2)
            sum_from_p = from_p1 + from_p2

            assert sum_from_p >= min_num_rules
            assert sum_from_p <= max_num_rules

            if sum_from_p == min_num_rules or sum_from_p == max_num_rules:
                # rules have maybe been added or removed, so the distribution is different
                continue

            num_rules_from_p1.append(from_p1)
            num_rules_from_p2.append(from_p2)

        # check mean and std
        if len(num_rules_from_p1) == 0:
            num_rules_from_p1 = [0]
        if len(num_rules_from_p2) == 0:
            num_rules_from_p2 = [0]


        mean_p1, std_p1 = np.mean(num_rules_from_p1), np.std(num_rules_from_p1)
        mean_p2, std_p2 = np.mean(num_rules_from_p2), np.std(num_rules_from_p2)

        # the distribution should be uniform if no fix is applied (i.e. >= min_num_rules and <= max_num_rules)
        expected_mean_p1 = num_rules_p1 / 2
        expected_mean_p2 = num_rules_p2 / 2
        assert mean_p1 == pytest.approx(expected_mean_p1, rel=num_rules_p1/10)
        assert mean_p2 == pytest.approx(expected_mean_p2, rel=num_rules_p2/10)

        expected_std_p1 = np.sqrt(num_rules_p1**2 / 12)
        expected_std_p2 = np.sqrt(num_rules_p2**2 / 12)
        assert std_p1 == pytest.approx(expected_std_p1, rel=num_rules_p1/10)
        assert std_p2 == pytest.approx(expected_std_p2, rel=num_rules_p2/10)

    except ValueError as e:
        if num_rules_p1 > 0 and num_rules_p2 > 0:
            # if either one of those is null, it's normal to get an error
            raise e


def test_distribution_java():
    min_num_rules, max_num_rules = 1, 60
    crossover_probability = 0.9

    train, _ = get_a0_0_iris_train_test()
    random_gen = np.random.Generator(np.random.MT19937(seed=2022))

    knowledge = HomoTriangleKnowledgeFactory_2_3_4_5(train.get_num_dim()).create()
    antecedent_factory = AllCombinationAntecedentFactory(knowledge, random_gen)
    consequent_factory = LearningBasic(train)
    rule_builder = RuleBuilderBasic(antecedent_factory, consequent_factory, knowledge)

    classification = SingleWinnerRuleSelection()
    objectives = np.array([ErrorRate(train), NumRules()])
    michigan_solution_builder = MichiganSolutionBuilder(random_gen, len(objectives), 0, rule_builder)

    problem = PittsburghProblem(train.get_num_dim(), objectives, 0, train, michigan_solution_builder, classification)

    sol1_indices = np.array([
        [1, 2, 0, 8],
        [1, 0, 1, 4],
        [1, 0, 0, 4],
        [3, 0, 5, 2],
        [9, 0, 0, 8]
    ], dtype=np.int32)
    sol2_indices = np.array([
        [1, 3, 7, 0],
        [6, 0, 10, 0],
        [1, 3, 7, 3]
    ], dtype=np.int32)

    michigan_sols = np.empty(len(sol1_indices), dtype=object)
    for i, indices in enumerate(sol1_indices):
        michigan_sols[i] = create_michigan_sol(train, antecedent_indices=indices)

    sol1 = create_pittsburgh_sol(train, classification, michigan_sols)

    michigan_sols = np.empty(len(sol2_indices), dtype=object)
    for i, indices in enumerate(sol2_indices):
        michigan_sols[i] = create_michigan_sol(train, antecedent_indices=indices)

    sol2 = create_pittsburgh_sol(train, classification, michigan_sols)

    pop = Population.new(X=np.array([[sol1], [sol2]], dtype=object))
    parents = np.array([[0, 1]])

    problem.evaluate(pop.get("X"))

    crossover = PittsburghCrossover(min_num_rules, max_num_rules, random_gen, prob=crossover_probability)
    offspring = crossover.do(problem, pop, parents=parents)

    problem.evaluate(offspring.get("X"))

    print("Parents")
    print(f"Parent 1:")
    for i, rule in enumerate(pop[0].X[0].get_vars()):
        print(f"Rule {i+1}: {rule}")

    print(f"\nParent 2:")
    for i, rule in enumerate(pop[1].X[0].get_vars()):
        print(f"Rule {i+1}: {rule}")

    print("\nOffspring:")
    for i, child in enumerate(offspring):
        print(f"Child {i+1}:")
        for j, rule in enumerate(child.X[0].get_vars()):
            print(f"Rule {j+1}: {rule}")

    # TODO: run multiple times and compare distribution of fitness, ... with the Java code