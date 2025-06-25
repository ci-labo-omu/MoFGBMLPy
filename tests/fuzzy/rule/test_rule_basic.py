import copy

import numpy as np

from mofgbmlpy.data.class_label.class_label_basic import ClassLabelBasic
from mofgbmlpy.fuzzy.knowledge.factory.homo_triangle_knowledge_factory_2_3_4_5 import (
    HomoTriangleKnowledgeFactory_2_3_4_5,
)
from mofgbmlpy.fuzzy.rule.antecedent.antecedent import Antecedent
from mofgbmlpy.fuzzy.rule.consequent.consequent_basic import ConsequentBasic

from mofgbmlpy.fuzzy.rule.consequent.ruleWeight.rule_weight_basic import RuleWeightBasic
from mofgbmlpy.fuzzy.rule.rule_basic import RuleBasic

from mofgbmlpy.fuzzy.rule.consequent.learning.learning_basic import LearningBasic
from util import get_a0_0_iris_train_test, get_a0_0_pima_train_test
import pytest

def test_deep_copy():
    # Just check if it raises an exception
    knowledge = HomoTriangleKnowledgeFactory_2_3_4_5(3).create()
    antecedent = Antecedent(np.array([0, 2, 1]), knowledge)
    consequent = ConsequentBasic(ClassLabelBasic(1), RuleWeightBasic(1))

    obj = RuleBasic(antecedent, consequent)
    _ = copy.deepcopy(obj)

    assert True

def test_example_from_java_iris():
    train, _ = get_a0_0_iris_train_test()
    knowledge = HomoTriangleKnowledgeFactory_2_3_4_5(train.get_num_dim()).create()
    antecedent = Antecedent(np.array([3, 0, 5, 2], np.int32), knowledge)
    consequent_factory = LearningBasic(train)
    consequent = consequent_factory.learning(antecedent)
    rule = RuleBasic(antecedent, consequent)

    pattern_id = 74
    p = train.get_pattern(pattern_id)

    expected_compatibility_grades = [0.05555558204650879, 1.0, 0.38983047008514404, 0.625]

    compatible_grades = antecedent.get_membership_values(p.get_attributes_vector())
    for i in range(len(expected_compatibility_grades)):
        assert pytest.approx(compatible_grades[i], 1e-14) == expected_compatibility_grades[i]

    prd = 1
    for i in range(len(compatible_grades)):
        prd *= compatible_grades[i]

    assert prd == pytest.approx(prd, 1e-14) == 0.013535786665652694

    compatible_grade = antecedent.get_compatible_grade_value_py(p.get_attributes_vector())
    assert prd == pytest.approx(compatible_grade, 1e-14)

    assert pytest.approx(compatible_grade, 1e-14) == 0.013535786665652694

    assert rule.get_class_label().get_class_label_value() == 2

    expected_confidence = [0.0, 0.33568177871190535, 0.6643182212880947]
    confidences = consequent_factory.calc_confidence_py(antecedent)
    assert len(confidences) == len(expected_confidence)
    for i in range(len(confidences)):
        assert pytest.approx(confidences[i], 1e-14) == expected_confidence[i], f"Confidence mismatch at index {i}: expected {expected_confidence[i]}, got {confidences[i]}"


    assert pytest.approx(rule.get_rule_weight_py().get_value(), 1e-14) == 0.3286364425761894


def test_example_from_java_pimas():
    train, _ = get_a0_0_pima_train_test()
    knowledge = HomoTriangleKnowledgeFactory_2_3_4_5(train.get_num_dim()).create()
    antecedent = Antecedent(np.array([9, 0, 0, 8, 0, 0, 0, 0], np.int32), knowledge)
    consequent_factory = LearningBasic(train)
    consequent = consequent_factory.learning(antecedent)
    rule = RuleBasic(antecedent, consequent)

    pattern_id = 66
    p = train.get_pattern(pattern_id)

    expected_compatibility_grades = [0.2941177785396576, 1.0, 1.0, 0.6363637447357178, 1.0, 1.0, 1.0, 1.0]

    compatible_grades = antecedent.get_membership_values(p.get_attributes_vector())
    for i in range(len(expected_compatibility_grades)):
        assert pytest.approx(compatible_grades[i], 1e-14) == expected_compatibility_grades[i]

    prd = 1
    for i in range(len(compatible_grades)):
        prd *= compatible_grades[i]

    assert prd == pytest.approx(prd, 1e-14) == 0.18716589094484704

    compatible_grade = antecedent.get_compatible_grade_value_py(p.get_attributes_vector())
    assert prd == pytest.approx(compatible_grade, 1e-14)

    assert pytest.approx(compatible_grade, 1e-14) == 0.18716589094484704

    assert rule.get_class_label().get_class_label_value() == 1

    expected_confidence = [0.4498382669189968, 0.5501617330810031]
    confidences = consequent_factory.calc_confidence_py(antecedent)
    assert len(confidences) == len(expected_confidence)
    for i in range(len(confidences)):
        assert pytest.approx(confidences[i], 1e-14) == expected_confidence[i], f"Confidence mismatch at index {i}: expected {expected_confidence[i]}, got {confidences[i]}"

    assert pytest.approx(rule.get_rule_weight_py().get_value(), 1e-14) == 0.10032346616200627
