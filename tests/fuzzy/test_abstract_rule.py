import numpy as np
import pytest
import xml.etree.cElementTree as xml_tree
from xml.dom import minidom
from mofgbmlpy.data.class_label.class_label_basic import ClassLabelBasic
from mofgbmlpy.fuzzy.knowledge.homo_triangle_knowledge_factory_2_3_4_5 import HomoTriangleKnowledgeFactory_2_3_4_5
from mofgbmlpy.fuzzy.knowledge.knowledge import Knowledge
from mofgbmlpy.fuzzy.rule.antecedent.antecedent import Antecedent
from mofgbmlpy.fuzzy.rule.consequent.consequent import Consequent
from mofgbmlpy.fuzzy.rule.consequent.ruleWeight.rule_weight_basic import RuleWeightBasic
from mofgbmlpy.fuzzy.rule.rule_basic import RuleBasic


def test_to_xml():
    knowledge = HomoTriangleKnowledgeFactory_2_3_4_5(3).create()
    antecedent = Antecedent(np.array([0, 1, 2]), knowledge)
    consequent = Consequent(ClassLabelBasic(1), RuleWeightBasic(0.3))
    rule = RuleBasic(antecedent, consequent)

    reparsed = minidom.parseString(xml_tree.tostring(rule.to_xml()))
    pretty_xml = reparsed.toprettyxml(indent="  ")
    print(pretty_xml)
    assert True


