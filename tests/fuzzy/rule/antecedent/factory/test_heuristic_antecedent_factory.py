import numpy as np
import pytest

from mofgbmlpy.data.class_label.class_label_basic import ClassLabelBasic
from mofgbmlpy.data.dataset import Dataset
from mofgbmlpy.data.input import Input
from mofgbmlpy.data.pattern import Pattern
from mofgbmlpy.fuzzy.knowledge.factory.homo_triangle_knowledge_factory_2_3_4_5 import \
    HomoTriangleKnowledgeFactory_2_3_4_5
from mofgbmlpy.fuzzy.knowledge.knowledge import Knowledge
from mofgbmlpy.fuzzy.rule.antecedent.factory.heuristic_antecedent_factory import HeuristicAntecedentFactory
from mofgbmlpy.main.arguments import Arguments


# (self, training_set, knowledge, is_dc_probability, dc_rate, antecedent_num_not_dont_care):

def get_training_set():
    args = Arguments()
    args.set("TRAIN_FILE", "../dataset/iris/a0_0_iris-10tra.dat")
    args.set("TEST_FILE", "../dataset/iris/a0_0_iris-10tra.dat")
    args.set("IS_MULTI_LABEL", False)
    train, _ = Input.get_train_test_files(args)
    return train


def test_none_knowledge():
    training_set = get_training_set()
    knowledge = None
    is_dc_probability = True
    dc_rate = 0.5
    antecedent_num_not_dont_care = 1

    with pytest.raises(Exception):
        HeuristicAntecedentFactory(training_set, knowledge, is_dc_probability, dc_rate, antecedent_num_not_dont_care)


def test_no_fuzzy_vars_knowledge():
    training_set = get_training_set()
    knowledge = Knowledge()
    is_dc_probability = True
    dc_rate = 0.5
    antecedent_num_not_dont_care = 1

    with pytest.raises(Exception):
        HeuristicAntecedentFactory(training_set, knowledge, is_dc_probability, dc_rate, antecedent_num_not_dont_care)


def test_none_training_set():
    training_set = None
    knowledge = HomoTriangleKnowledgeFactory_2_3_4_5(1).create()
    is_dc_probability = True
    dc_rate = 0.5
    antecedent_num_not_dont_care = 1
    with pytest.raises(Exception):
        HeuristicAntecedentFactory(training_set, knowledge, is_dc_probability, dc_rate, antecedent_num_not_dont_care)


def test_different_num_dim_training_set():
    training_set = get_training_set()
    knowledge = HomoTriangleKnowledgeFactory_2_3_4_5(1).create()
    is_dc_probability = True
    dc_rate = 0.5
    antecedent_num_not_dont_care = 1

    with pytest.raises(Exception):
        HeuristicAntecedentFactory(training_set, knowledge, is_dc_probability, dc_rate, antecedent_num_not_dont_care)


def test_is_dc_probability_none():
    training_set = get_training_set()
    knowledge = HomoTriangleKnowledgeFactory_2_3_4_5(training_set.get_num_dim()).create()
    is_dc_probability = None  # will be converted to False
    dc_rate = 0.5
    antecedent_num_not_dont_care = 1

    HeuristicAntecedentFactory(training_set, knowledge, is_dc_probability, dc_rate, antecedent_num_not_dont_care)


@pytest.mark.parametrize("value", np.random.uniform(-1, 1, 5))
def test_dc_rate(value):
    training_set = get_training_set()
    knowledge = HomoTriangleKnowledgeFactory_2_3_4_5(training_set.get_num_dim()).create()
    is_dc_probability = True
    dc_rate = value
    antecedent_num_not_dont_care = 1

    if dc_rate < 0 or dc_rate > 1:
        with pytest.raises(Exception):
            HeuristicAntecedentFactory(training_set, knowledge, is_dc_probability, dc_rate, antecedent_num_not_dont_care)
    else:
        HeuristicAntecedentFactory(training_set, knowledge, is_dc_probability, dc_rate, antecedent_num_not_dont_care)


def test_antecedent_num_not_dont_care_negative():
    training_set = get_training_set()
    knowledge = HomoTriangleKnowledgeFactory_2_3_4_5(training_set.get_num_dim()).create()
    is_dc_probability = True
    dc_rate = 0.5
    antecedent_num_not_dont_care = -1

    with pytest.raises(Exception):
        HeuristicAntecedentFactory(training_set, knowledge, is_dc_probability, dc_rate, antecedent_num_not_dont_care)

def create_example(is_dc_probability=True, dc_rate=0.5, antecedent_num_not_dont_care=1):
    training_set = get_training_set()
    knowledge = HomoTriangleKnowledgeFactory_2_3_4_5(training_set.get_num_dim()).create()
    return HeuristicAntecedentFactory(training_set, knowledge, is_dc_probability, dc_rate, antecedent_num_not_dont_care)


def test_calculate_antecedent_part_none_pattern():
    factory = create_example()
    pattern = None

    with pytest.raises(Exception):
        factory.calculate_antecedent_part_py(pattern)


def test_calculate_antecedent_part_empty_pattern():
    factory = create_example()
    pattern = Pattern(0, np.empty(0), ClassLabelBasic(0))

    with pytest.raises(Exception):
        factory.calculate_antecedent_part_py(pattern)


def test_calculate_antecedent_part_different_dimension_than_knowledge():
    factory = create_example()
    pattern = Pattern(0, np.array([1.0]), ClassLabelBasic(0))

    with pytest.raises(Exception):
        factory.calculate_antecedent_part_py(pattern)


def test_calculate_antecedent_part_categorical_attribute():
    factory = create_example(dc_rate=0)
    pattern = Pattern(0, np.array([-1.0, -2.0, -3.0, -4.0]), ClassLabelBasic(0))

    antecedent_indices = factory.calculate_antecedent_part_py(pattern)
    for i in range(len(antecedent_indices)):
        assert antecedent_indices[i] == pattern.get_attribute_value(i)


def test_calculate_antecedent_part_all_dc_from_dc_rate():
    factory = create_example(dc_rate=1)
    pattern = Pattern(0, np.array([1.0, 2.0, 3.0, 4.0]), ClassLabelBasic(0))

    antecedent_indices = factory.calculate_antecedent_part_py(pattern)
    for i in range(len(antecedent_indices)):
        assert antecedent_indices[i] == 0.0


def test_calculate_antecedent_part_all_dc_from_not_dc_probability():
    factory = create_example(antecedent_num_not_dont_care=0, is_dc_probability=False)
    pattern = Pattern(0, np.array([1.0, 2.0, 3.0, 4.0]), ClassLabelBasic(0))

    antecedent_indices = factory.calculate_antecedent_part_py(pattern)
    for i in range(len(antecedent_indices)):
        assert antecedent_indices[i] == 0.0


def test_calculate_antecedent_part():
    factory = create_example()
    pattern = Pattern(0, np.array([1.0, 2.0, 3.0, 4.0]), ClassLabelBasic(0))

    with pytest.raises(Exception):
        factory.calculate_antecedent_part(pattern)