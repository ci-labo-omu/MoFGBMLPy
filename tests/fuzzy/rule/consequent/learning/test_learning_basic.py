import copy

import numpy as np
import pytest

from mofgbmlpy.data.class_label.class_label_basic import ClassLabelBasic
from mofgbmlpy.data.dataset import Dataset
from mofgbmlpy.data.input import Input
from mofgbmlpy.data.pattern import Pattern
from mofgbmlpy.fuzzy.knowledge.factory.homo_triangle_knowledge_factory_2_3_4_5 import \
    HomoTriangleKnowledgeFactory_2_3_4_5
from mofgbmlpy.fuzzy.rule.antecedent.antecedent import Antecedent
from mofgbmlpy.fuzzy.rule.consequent.learning.learning_basic import LearningBasic
from mofgbmlpy.main.arguments.arguments import Arguments
from util import get_a0_0_iris_train_test, float_eq

train, _ = get_a0_0_iris_train_test()


def test_none_training_set():
    with pytest.raises(TypeError):
        LearningBasic(None)


def test_learning_none_antecedent():
    learner = LearningBasic(train)
    with pytest.raises(TypeError):
        learner.learning(None)


def test_calc_confidence_none_antecedent():
    learner = LearningBasic(train)
    with pytest.raises(TypeError):
        learner.calc_confidence_py(None)


def test_calc_confidence_antecedent_iris():
    learner = LearningBasic(train)
    antecedent = Antecedent(np.array([0, 1, 2, 3]), knowledge=HomoTriangleKnowledgeFactory_2_3_4_5(4).create())
    confidence = learner.calc_confidence_py(antecedent)

    assert float_eq(confidence[0], 0.5602049144376764) and float_eq(confidence[1], 0.4397950855623236) and float_eq(confidence[2], 0.0)


def test_calc_confidence_custom_dataset():
    dataset = Dataset(4, 4, 3, np.array([
        Pattern(0, np.array([0.5, 0.0, 1.0, 0.0]), ClassLabelBasic(0)),
        Pattern(1, np.array([0.0, 0.5, 0.0, 1.0]), ClassLabelBasic(1)),
        Pattern(2, np.array([1.0, 0.5, 0.0, 0.0]), ClassLabelBasic(2)),
        Pattern(3, np.array([0.5, 1.0, 0.0, 0.0]), ClassLabelBasic(2)),
    ]))

    learner = LearningBasic(dataset)
    antecedent = Antecedent(np.array([1, 0, 0, 0]), knowledge=HomoTriangleKnowledgeFactory_2_3_4_5(4).create())
    confidence = learner.calc_confidence_py(antecedent)

    expected = [0.25, 0.5, 0.25]

    for i in range(len(confidence)):
        assert confidence[i] == expected[i]


def test_calc_confidence_all_zero():
    dataset = Dataset(4, 4, 3, np.array([
        Pattern(0, np.array([0.5, 0.0, 1.0, 0.0]), ClassLabelBasic(0)),
        Pattern(1, np.array([0.0, 0.5, 0.0, 1.0]), ClassLabelBasic(1)),
        Pattern(2, np.array([0.0, 0.5, 0.0, 0.0]), ClassLabelBasic(2)),
        Pattern(3, np.array([0.5, 1.0, 0.0, 0.0]), ClassLabelBasic(2)),
    ]))

    learner = LearningBasic(dataset)
    antecedent = Antecedent(np.array([14, 0, 0, 0]), knowledge=HomoTriangleKnowledgeFactory_2_3_4_5(4).create())
    confidence = learner.calc_confidence_py(antecedent)

    for i in range(len(confidence)):
        assert confidence[i] == 0.0


def test_calc_class_label_none_confidence():
    learner = LearningBasic(train)
    with pytest.raises(TypeError):
        learner.calc_class_label(None)


def test_calc_class_label_empty():
    confidence = np.empty(0)
    learner = LearningBasic(train)
    assert learner.calc_class_label(confidence).get_class_label_value() == -1


def test_calc_class_label_same_highest():
    confidence = np.array([0.3, 0.2, 0.3, 0.2])
    learner = LearningBasic(train)
    label = learner.calc_class_label(confidence)
    assert label.is_rejected() and label.get_class_label_value() == -1


def test_calc_class_label_different_highest():
    confidence = np.array([0.3, 0.0, 0.5, 0.2])
    learner = LearningBasic(train)
    label = learner.calc_class_label(confidence)
    assert not label.is_rejected() and label.get_class_label_value() == 2


def test_calc_rule_weight_none_class_label():
    confidence = np.array([0.4, 0.0, 0.5, 0.2])
    reject_threshold = 0
    learner = LearningBasic(train)

    with pytest.raises(TypeError):
        learner.calc_rule_weight(None, confidence, reject_threshold)


def test_calc_rule_weight_none_confidence():
    class_label = ClassLabelBasic(0)
    reject_threshold = 0
    learner = LearningBasic(train)

    with pytest.raises(TypeError):
        learner.calc_rule_weight(class_label, None, reject_threshold)


def test_calc_rule_weight_empty_confidence():
    class_label = ClassLabelBasic(0)
    reject_threshold = 0
    learner = LearningBasic(train)

    with pytest.raises(IndexError):
        learner.calc_rule_weight(class_label, np.empty(0), reject_threshold)


def test_calc_rule_weight_none_reject_threshold():
    class_label = ClassLabelBasic(0)
    confidence = np.array([0.4, 0.0, 0.5, 0.2])
    learner = LearningBasic(train)

    with pytest.raises(TypeError):
        learner.calc_rule_weight(class_label, confidence, None)


def test_calc_rule_weight_incompatible_class_label_and_confidence():
    class_label = ClassLabelBasic(4)
    confidence = np.array([0.4, 0.1, 0.3, 0.2])
    reject_threshold = 0
    learner = LearningBasic(train)

    with pytest.raises(IndexError):
        learner.calc_rule_weight(class_label, confidence, reject_threshold)


def test_calc_rule_weight_confidence_below_0_5():
    class_label = ClassLabelBasic(0)
    confidence = np.array([0.4, 0.1, 0.3, 0.2])
    reject_threshold = 0
    learner = LearningBasic(train)

    rule_weight = learner.calc_rule_weight(class_label, confidence, reject_threshold)
    assert rule_weight.get_value() == 0 and class_label.is_rejected()


def test_calc_rule_weight_equal_0_5():
    class_label = ClassLabelBasic(0)
    confidence = np.array([0.5, 0.1, 0.21, 0.19])
    reject_threshold = 0
    learner = LearningBasic(train)

    rule_weight = learner.calc_rule_weight(class_label, confidence, reject_threshold)
    assert rule_weight.get_value() == 0 and class_label.is_rejected()


def test_calc_rule_weight_above_0_5():
    class_label = ClassLabelBasic(0)
    confidence = np.array([0.6, 0.1, 0.19, 0.11])
    reject_threshold = 0
    learner = LearningBasic(train)

    rule_weight = learner.calc_rule_weight(class_label, confidence, reject_threshold)

    # Taken from the Java version
    assert float_eq(rule_weight.get_value(), 0.2)


def test_calc_rule_weight_equals_threshold():
    class_label = ClassLabelBasic(0)
    confidence = np.array([0.6, 0.1, 0.19, 0.11])
    reject_threshold = 0.2
    learner = LearningBasic(train)

    rule_weight = learner.calc_rule_weight(class_label, confidence, reject_threshold)

    # Taken from the Java version
    assert rule_weight.get_value() == 0.0 and class_label.is_rejected()


def test_calc_rule_weight_below_threshold():
    class_label = ClassLabelBasic(0)
    confidence = np.array([0.6, 0.1, 0.19, 0.11])
    reject_threshold = 0.3
    learner = LearningBasic(train)

    rule_weight = learner.calc_rule_weight(class_label, confidence, reject_threshold)

    # Taken from the Java version
    assert rule_weight.get_value() == 0.0 and class_label.is_rejected()


def test_eq_same():
    assert LearningBasic(train) == LearningBasic(train)


def test_eq_different():
    dataset = Dataset(2, 4, 3, np.array([
        Pattern(0, np.array([0.5, 0.0, 1.0, 0.0]), ClassLabelBasic(0)),
        Pattern(1, np.array([0.0, 0.5, 0.0, 1.0]), ClassLabelBasic(1)),
    ]))
    assert LearningBasic(train) != LearningBasic(dataset)


def test_deep_copy():
    obj = LearningBasic(train)
    obj_copy = copy.deepcopy(obj)

    assert obj == obj_copy
