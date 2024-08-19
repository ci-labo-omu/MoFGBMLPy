import copy

import numpy as np
import pytest

from mofgbmlpy.data.class_label.class_label_multi import ClassLabelMulti
from mofgbmlpy.data.dataset import Dataset
from mofgbmlpy.data.input import Input
from mofgbmlpy.data.pattern import Pattern
from mofgbmlpy.fuzzy.knowledge.factory.homo_triangle_knowledge_factory_2_3_4_5 import \
    HomoTriangleKnowledgeFactory_2_3_4_5
from mofgbmlpy.fuzzy.rule.antecedent.antecedent import Antecedent
from mofgbmlpy.fuzzy.rule.consequent.learning.learning_multi import LearningMulti
from mofgbmlpy.main.arguments import Arguments
from util import float_eq, get_a0_0_german_train_test

train, _ = get_a0_0_german_train_test()


def test_none_training_set():
    with pytest.raises(TypeError):
        LearningMulti(None)


def test_learning_none_antecedent():
    learner = LearningMulti(train)
    with pytest.raises(TypeError):
        learner.learning(None)


def test_calc_confidence_none_antecedent():
    learner = LearningMulti(train)
    with pytest.raises(TypeError):
        learner.calc_confidence_py(None)


def test_calc_confidence_antecedent_german():
    learner = LearningMulti(train)
    antecedent = Antecedent(np.array([-1, 0, -2, -1, 1, -2,
                                      -2, 2, -2, -1, 3, -4,
                                      4, -2, -1, 5, -1, 0,
                                      -2, -1]), knowledge=HomoTriangleKnowledgeFactory_2_3_4_5(train.get_num_dim()).create())
    confidence = learner.calc_confidence_py(antecedent)

    assert confidence[0][0] == 0.0 and confidence[0][1] == 0.0 and confidence[1][0] == 0.0 and confidence[1][1] == 0.0


def test_calc_confidence_custom_dataset():
    dataset = Dataset(4, 4, 3, np.array([
        Pattern(0, np.array([0.5, 0.0, 1.0, 0.0]), ClassLabelMulti(np.array([1, 0, 0]))),
        Pattern(1, np.array([0.0, 0.5, 0.0, 1.0]), ClassLabelMulti(np.array([0, 1, 0]))),
        Pattern(2, np.array([1.0, 0.5, 0.0, 0.0]), ClassLabelMulti(np.array([1, 0, 1]))),
        Pattern(3, np.array([0.5, 1.0, 0.0, 0.0]), ClassLabelMulti(np.array([1, 0, 0]))),
    ]))

    learner = LearningMulti(dataset)
    antecedent = Antecedent(np.array([1, 0, 0, 0]), knowledge=HomoTriangleKnowledgeFactory_2_3_4_5(4).create())
    confidence = learner.calc_confidence_py(antecedent)

    expected = [[0.5, 0.5],
                [0.5, 0.5],
                [1.0, 0.0]]  # Taken from the Java version

    for i in range(len(confidence)):
        assert confidence[i][0] == expected[i][0]
        assert confidence[i][1] == expected[i][1]

def test_calc_confidence_all_zero():
    dataset = Dataset(4, 4, 3, np.array([
        Pattern(0, np.array([0.5, 0.0, 1.0, 0.0]), ClassLabelMulti(np.array([1, 0, 0]))),
        Pattern(1, np.array([0.0, 0.5, 0.0, 1.0]), ClassLabelMulti(np.array([0, 1, 0]))),
        Pattern(2, np.array([0.0, 0.5, 0.0, 0.0]), ClassLabelMulti(np.array([1, 0, 1]))),
        Pattern(3, np.array([0.5, 1.0, 0.0, 0.0]), ClassLabelMulti(np.array([1, 0, 0]))),
    ]))

    learner = LearningMulti(dataset)
    antecedent = Antecedent(np.array([14, 0, 0, 0]), knowledge=HomoTriangleKnowledgeFactory_2_3_4_5(4).create())
    confidence = learner.calc_confidence_py(antecedent)

    for i in range(len(confidence)):
        assert confidence[i][0] == 0.0
        assert confidence[i][1] == 0.0


def test_calc_class_label_none_confidence():
    learner = LearningMulti(train)
    with pytest.raises(TypeError):
        learner.calc_class_label(None)


def test_calc_class_label_same_highest():
    confidence = np.array([[0.5, 0.5], [0.1, 0.9]])
    learner = LearningMulti(train)
    label = learner.calc_class_label(confidence)
    assert label.is_rejected()


def test_calc_class_label_different_highest():
    confidence = np.array([[0.6, 0.4], [0.1, 0.9]])
    learner = LearningMulti(train)
    label = learner.calc_class_label(confidence)
    assert not label.is_rejected()
    value = label.get_class_label_value()
    assert value[0] == 0
    assert value[1] == 1


def test_calc_rule_weight_none_class_label():
    confidence = np.array([[0.6, 0.4], [0.1, 0.9]])
    reject_threshold = 0
    learner = LearningMulti(train)

    with pytest.raises(TypeError):
        learner.calc_rule_weight(None, confidence, reject_threshold)


def test_calc_rule_weight_none_confidence():
    class_label = ClassLabelMulti(np.array([0, 1]))
    reject_threshold = 0
    learner = LearningMulti(train)

    with pytest.raises(TypeError):
        learner.calc_rule_weight(class_label, None, reject_threshold)


def test_calc_rule_weight_empty_confidence():
    class_label = ClassLabelMulti(np.array([0, 1]))
    reject_threshold = 0
    learner = LearningMulti(train)

    with pytest.raises(Exception):
        learner.calc_rule_weight(class_label, np.empty(0), reject_threshold)


def test_calc_rule_weight_none_reject_threshold():
    class_label = ClassLabelMulti(np.array([0, 1]))
    confidence = np.array([[0.6, 0.4], [0.1, 0.9]])
    learner = LearningMulti(train)

    with pytest.raises(TypeError):
        learner.calc_rule_weight(class_label, confidence, None)


def test_eq_same():
    assert LearningMulti(train) == LearningMulti(train)


def test_eq_different():
    dataset = Dataset(2, 4, 3, np.array([
        Pattern(0, np.array([0.5, 0.0, 1.0, 0.0]), ClassLabelMulti(np.array([1, 0, 0]))),
        Pattern(1, np.array([0.0, 0.5, 0.0, 1.0]), ClassLabelMulti(np.array([0, 1, 0]))),
    ]))
    assert LearningMulti(train) != LearningMulti(dataset)


def test_deep_copy():
    obj = LearningMulti(train)
    obj_copy = copy.deepcopy(obj)

    assert obj == obj_copy
