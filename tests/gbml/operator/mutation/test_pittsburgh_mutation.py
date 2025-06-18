# TODO: The Java version is not consistent (mutation rate param is not used), so this test won't pass if the Python version is like the Java one


# import copy
# import json
# import os
# from pathlib import Path
#
# import numpy as np
# from mofgbmlpy.gbml.operator.crossover.uniform_crossover_single_offspring_michigan import (
#     UniformCrossoverSingleOffspringMichigan,
# )
#
# from mofgbmlpy.fuzzy.knowledge.factory.homo_triangle_knowledge_factory_2_3_4_5 import (
#     HomoTriangleKnowledgeFactory_2_3_4_5,
# )
#
# from mofgbmlpy.fuzzy.rule.antecedent.factory.heuristic_antecedent_factory import HeuristicAntecedentFactory
#
# from mofgbmlpy.fuzzy.rule.consequent.learning.learning_basic import LearningBasic
#
# from mofgbmlpy.fuzzy.rule.rule_builder_basic import RuleBuilderBasic
#
# from mofgbmlpy.gbml.solution.michigan_solution_builder import MichiganSolutionBuilder
#
# from mofgbmlpy.gbml.problem.michigan_problem import MichiganProblem
# from pymoo.core.population import Population
#
# from mofgbmlpy.gbml.operator.mutation.michigan_mutation import MichiganMutation
#
# from mofgbmlpy.gbml.operator.mutation.pittsburgh_mutation import PittsburghMutation
#
# from mofgbmlpy.fuzzy.classification.single_winner_rule_selection import SingleWinnerRuleSelection
#
# from mofgbmlpy.gbml.problem.pittsburgh_problem import PittsburghProblem
#
# from mofgbmlpy.gbml.objectives.pittsburgh.error_rate import ErrorRate
#
# from mofgbmlpy.gbml.objectives.pittsburgh.num_rules import NumRules
# from util import get_a0_0_iris_train_test, create_pittsburgh_sol, create_michigan_sol
# import pytest
#
#
# def get_config(mutation_rt):
#     train, _ = get_a0_0_iris_train_test()
#     random_gen = np.random.Generator(np.random.MT19937(seed=2022))
#     knowledge = HomoTriangleKnowledgeFactory_2_3_4_5(train.get_num_dim()).create()
#
#     mutation = PittsburghMutation(knowledge, random_gen, mutation_rt)
#     classification = SingleWinnerRuleSelection()
#
#     antecedent_factory = HeuristicAntecedentFactory(train, knowledge, False, 0.8, 5, random_gen)
#     consequent_factory = LearningBasic(train)
#     rule_builder = RuleBuilderBasic(antecedent_factory, consequent_factory, knowledge)
#
#     objectives = np.array([ErrorRate(train), NumRules()])
#     michigan_solution_builder = MichiganSolutionBuilder(random_gen, len(objectives), 0, rule_builder)
#
#     tests_root = Path(__file__).parents[3]
#     test_population_path = os.path.join(tests_root, "test_data", "population_samples", "iris.json")
#
#     indices_pop = np.array(json.load(open(test_population_path, "r"))[0], int)
#
#     michigan_sols = np.array([create_michigan_sol(train, antecedent_indices=indices) for indices in indices_pop], object)
#     sol = create_pittsburgh_sol(train, classification, michigan_sols, michigan_solution_builder)
#
#     problem = PittsburghProblem(train.get_num_dim(), objectives, 0, train, michigan_solution_builder, classification)
#
#     num_iters = 10000
#
#     return problem, sol, num_iters, mutation
#
#
# def test_mutation_rt_1():
#     mutation_rt = 1
#     problem, sol, num_iters, mutation = get_config(mutation_rt)
#
#     offsprings = np.empty(num_iters, dtype=object)
#
#     for i in range(num_iters):
#         pop = Population.new(X=np.array([[copy.deepcopy(sol)]]))
#         new_sol = mutation.do(problem, pop)[0].X[0]
#         assert sol != new_sol, f"Mutation did not change the solution"
#
#
# def test_mutation_rt_0():
#     mutation_rt = 0
#     problem, sol, num_iters, mutation = get_config(mutation_rt)
#
#     for i in range(num_iters):
#         pop = Population.new(X=np.array([[copy.deepcopy(sol)]]))
#         new_sol = mutation.do(problem, pop)[0].X[0]
#
#         assert sol == new_sol, f"Mutation did change the solution"
