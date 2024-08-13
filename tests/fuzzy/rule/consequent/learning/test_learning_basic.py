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
from mofgbmlpy.main.arguments import Arguments
from util import get_a0_0_iris_train_test, float_eq

train, _ = get_a0_0_iris_train_test()


def test_none_training_set():
    with pytest.raises(Exception):
        LearningBasic(None)


def test_learning_none_antecedent():
    learner = LearningBasic(train)
    with pytest.raises(Exception):
        learner.learning(None)


def test_calc_confidence_none_antecedent():
    learner = LearningBasic(train)
    with pytest.raises(Exception):
        learner.calc_confidence(None)


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


def test_calc_confidence_():
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
    with pytest.raises(Exception):
        learner.calc_class_label(None)





def test_deep_copy():

    # Just check if it raises an exception
    obj = LearningBasic(train)
    _ = copy.deepcopy(obj)

    assert True