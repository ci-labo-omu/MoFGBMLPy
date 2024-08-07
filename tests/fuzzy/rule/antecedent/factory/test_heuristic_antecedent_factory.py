import copy

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
from util import get_a0_0_iris_train_test

random_gen = np.random.Generator(np.random.MT19937(seed=2022))
training_set, _ = get_a0_0_iris_train_test()

def test_none_knowledge():
    knowledge = None
    is_dc_probability = True
    dc_rate = 0.5
    antecedent_number_do_not_dont_care = 1

    with pytest.raises(Exception):
        HeuristicAntecedentFactory(training_set, knowledge, is_dc_probability, dc_rate, antecedent_number_do_not_dont_care, random_gen)


def test_no_fuzzy_vars_knowledge():
    knowledge = Knowledge()
    is_dc_probability = True
    dc_rate = 0.5
    antecedent_number_do_not_dont_care = 1

    with pytest.raises(Exception):
        HeuristicAntecedentFactory(training_set, knowledge, is_dc_probability, dc_rate, antecedent_number_do_not_dont_care, random_gen)


def test_none_training_set():
    knowledge = HomoTriangleKnowledgeFactory_2_3_4_5(1).create()
    is_dc_probability = True
    dc_rate = 0.5
    antecedent_number_do_not_dont_care = 1
    with pytest.raises(Exception):
        HeuristicAntecedentFactory(training_set, knowledge, is_dc_probability, dc_rate, antecedent_number_do_not_dont_care, random_gen)


def test_different_num_dim_training_set():
    knowledge = HomoTriangleKnowledgeFactory_2_3_4_5(1).create()
    is_dc_probability = True
    dc_rate = 0.5
    antecedent_number_do_not_dont_care = 1

    with pytest.raises(Exception):
        HeuristicAntecedentFactory(training_set, knowledge, is_dc_probability, dc_rate, antecedent_number_do_not_dont_care, random_gen)


def test_is_dc_probability_none():
    knowledge = HomoTriangleKnowledgeFactory_2_3_4_5(training_set.get_num_dim()).create()
    is_dc_probability = None  # will be converted to False
    dc_rate = 0.5
    antecedent_number_do_not_dont_care = 1

    HeuristicAntecedentFactory(training_set, knowledge, is_dc_probability, dc_rate, antecedent_number_do_not_dont_care, random_gen)


@pytest.mark.parametrize("value", np.random.uniform(-1, 1, 5))
def test_dc_rate(value):
    knowledge = HomoTriangleKnowledgeFactory_2_3_4_5(training_set.get_num_dim()).create()
    is_dc_probability = True
    dc_rate = value
    antecedent_number_do_not_dont_care = 1

    if dc_rate < 0 or dc_rate > 1:
        with pytest.raises(Exception):
            HeuristicAntecedentFactory(training_set, knowledge, is_dc_probability, dc_rate, antecedent_number_do_not_dont_care, random_gen)
    else:
        HeuristicAntecedentFactory(training_set, knowledge, is_dc_probability, dc_rate, antecedent_number_do_not_dont_care, random_gen)


def test_antecedent_number_do_not_dont_care_negative():
    knowledge = HomoTriangleKnowledgeFactory_2_3_4_5(training_set.get_num_dim()).create()
    is_dc_probability = True
    dc_rate = 0.5
    antecedent_number_do_not_dont_care = -1

    with pytest.raises(Exception):
        HeuristicAntecedentFactory(training_set, knowledge, is_dc_probability, dc_rate, antecedent_number_do_not_dont_care, random_gen)

def create_example(is_dc_probability=True, dc_rate=0.5, antecedent_number_do_not_dont_care=1):
    knowledge = HomoTriangleKnowledgeFactory_2_3_4_5(training_set.get_num_dim()).create()
    return HeuristicAntecedentFactory(training_set, knowledge, is_dc_probability, dc_rate, antecedent_number_do_not_dont_care, random_gen)


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
    factory = create_example(antecedent_number_do_not_dont_care=0, is_dc_probability=False)
    pattern = Pattern(0, np.array([1.0, 2.0, 3.0, 4.0]), ClassLabelBasic(0))

    antecedent_indices = factory.calculate_antecedent_part_py(pattern)
    for i in range(len(antecedent_indices)):
        assert antecedent_indices[i] == 0.0


def test_calculate_antecedent_part():
    factory = create_example()
    pattern = Pattern(0, np.array([1.0, 2.0, 3.0, 4.0]), ClassLabelBasic(0))

    with pytest.raises(Exception):
        factory.calculate_antecedent_part(pattern)


def test_deep_copy():
    # Just check if it raises an exception
    _ = copy.deepcopy(create_example(antecedent_number_do_not_dont_care=0, is_dc_probability=False))

    assert True