import numpy as np
import pytest
import xml.etree.cElementTree as xml_tree
from xml.dom import minidom
from mofgbmlpy.data.class_label.class_label_basic import ClassLabelBasic
from mofgbmlpy.data.input import Input
from mofgbmlpy.fuzzy.classifier.classification.single_winner_rule_selection import SingleWinnerRuleSelection
from mofgbmlpy.fuzzy.classifier.classifier import Classifier
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


def test_to_xml_run():
    # Only test if it doesn't return an exception
    args = MoFGBMLNSGAIIArgs()

    args.set("TRAIN_FILE", "../dataset/iris/a0_0_iris-10tra.dat")
    args.set("TEST_FILE", "../dataset/iris/a0_0_iris-10tra.dat")
    args.set("IS_MULTI_LABEL", False)

    training_data_set, _ = Input.get_train_test_files(args)
    knowledge = HomoTriangleKnowledgeFactory_2_3_4_5(training_data_set.get_num_dim()).create()
    rule_builder = RuleBuilderBasic(AllCombinationAntecedentFactory(knowledge), LearningBasic(training_data_set), knowledge)
    michigan_builder = MichiganSolutionBuilder(2, 0, rule_builder)
    classifier = Classifier(SingleWinnerRuleSelection())
    problem = PittsburghProblem(3, ["error-rate", "num-rules"], 0, training_data_set, michigan_builder, classifier)
    sol = problem.create_solution()
    reparsed = minidom.parseString(xml_tree.tostring(sol.to_xml()))
    _ = reparsed.toprettyxml(indent="  ")

    assert True


