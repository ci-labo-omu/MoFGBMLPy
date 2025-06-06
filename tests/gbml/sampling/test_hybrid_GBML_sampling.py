import copy

import numpy as np

from mofgbmlpy.data.input import Input
from mofgbmlpy.fuzzy.classification.single_winner_rule_selection import SingleWinnerRuleSelection
from mofgbmlpy.fuzzy.knowledge.factory.homo_triangle_knowledge_factory_2_3_4_5 import (
    HomoTriangleKnowledgeFactory_2_3_4_5,
)
from mofgbmlpy.fuzzy.rule.antecedent.factory.all_combination_antecedent_factory import AllCombinationAntecedentFactory
from mofgbmlpy.fuzzy.rule.consequent.learning.learning_basic import LearningBasic
from mofgbmlpy.fuzzy.rule.rule_builder_basic import RuleBuilderBasic
from mofgbmlpy.gbml.objectives.pittsburgh.error_rate import ErrorRate
from mofgbmlpy.gbml.objectives.pittsburgh.num_rules import NumRules
from mofgbmlpy.gbml.problem.michigan_problem import MichiganProblem
from mofgbmlpy.gbml.problem.pittsburgh_problem import PittsburghProblem
from mofgbmlpy.gbml.solution.michigan_solution_builder import MichiganSolutionBuilder

from mofgbmlpy.gbml.solution.pittsburgh_solution import PittsburghSolution
from mofgbmlpy.main.arguments.arguments import Arguments
from util import get_a0_0_iris_train_test
from mofgbmlpy.gbml.sampling.hybrid_GBML_sampling import HybridGBMLSampling


def test_sampling():
    random_gen = np.random.Generator(np.random.MT19937(seed=2022))
    train, _ = get_a0_0_iris_train_test()
    pop_size = 10

    knowledge = HomoTriangleKnowledgeFactory_2_3_4_5(train.get_num_dim()).create()
    antecedent_factory = AllCombinationAntecedentFactory(knowledge, random_gen)
    consequent_factory = LearningBasic(train)
    rule_builder = RuleBuilderBasic(antecedent_factory, consequent_factory, knowledge)
    objectives = np.array([ErrorRate(train), NumRules()])
    michigan_solution_builder = MichiganSolutionBuilder(random_gen, len(objectives), 0, rule_builder)
    classification = SingleWinnerRuleSelection()

    sampling = HybridGBMLSampling(consequent_factory)
    problem = PittsburghProblem(train.get_num_dim(), objectives, 0, train, michigan_solution_builder, classification)
    pop = sampling._do(problem, pop_size)

    assert len(pop) == pop_size
    assert len(pop[0]) == 1
    assert isinstance(pop[0][0], PittsburghSolution)
