import numpy as np
import pytest
import xml.etree.cElementTree as xml_tree
from xml.dom import minidom
from mofgbmlpy.data.class_label.class_label_basic import ClassLabelBasic
from mofgbmlpy.data.input import Input
from mofgbmlpy.fuzzy.classification.single_winner_rule_selection import SingleWinnerRuleSelection
from mofgbmlpy.fuzzy.knowledge.factory.homo_triangle_knowledge_factory_2_3_4_5 import HomoTriangleKnowledgeFactory_2_3_4_5
from mofgbmlpy.fuzzy.knowledge.knowledge import Knowledge
from mofgbmlpy.fuzzy.rule.antecedent.antecedent import Antecedent
from mofgbmlpy.fuzzy.rule.antecedent.factory.all_combination_antecedent_factory import AllCombinationAntecedentFactory
from mofgbmlpy.fuzzy.rule.consequent.consequent import Consequent
from mofgbmlpy.fuzzy.rule.consequent.learning.learning_basic import LearningBasic
from mofgbmlpy.fuzzy.rule.consequent.ruleWeight.rule_weight_basic import RuleWeightBasic
from mofgbmlpy.fuzzy.rule.rule_basic import RuleBasic
from mofgbmlpy.fuzzy.rule.rule_builder_basic import RuleBuilderBasic
from mofgbmlpy.gbml.problem.pittsburgh_problem import PittsburghProblem
from mofgbmlpy.gbml.solution.michigan_solution_builder import MichiganSolutionBuilder
from mofgbmlpy.main.nsgaii.mofgbml_nsgaii_args import MoFGBMLNSGAIIArgs
from util import get_a0_0_iris_train_test

training_data_set, _ = get_a0_0_iris_train_test()

def test_to_xml_run():
    random_gen = np.random.Generator(np.random.MT19937(seed=2022))
    # Only test if it doesn't return an exception
    knowledge = HomoTriangleKnowledgeFactory_2_3_4_5(training_data_set.get_num_dim()).create()
    rule_builder = RuleBuilderBasic(AllCombinationAntecedentFactory(knowledge, random_gen), LearningBasic(training_data_set), knowledge)
    michigan_builder = MichiganSolutionBuilder(random_gen, 2, 0, rule_builder)
    classification = SingleWinnerRuleSelection()
    problem = PittsburghProblem(3, ["error-rate", "num-rules"], 0, training_data_set, michigan_builder, classification)
    sol = problem.create_solution()
    reparsed = minidom.parseString(xml_tree.tostring(sol.to_xml()))
    _ = reparsed.toprettyxml(indent="  ")

    assert True


