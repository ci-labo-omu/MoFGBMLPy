import copy

import numpy as np
import pytest

from mofgbmlpy.data.class_label.class_label_basic import ClassLabelBasic
from mofgbmlpy.data.dataset import Dataset
from mofgbmlpy.data.input import Input
from mofgbmlpy.data.pattern import Pattern
from mofgbmlpy.exception.uninitialized_knowledge_exception import UninitializedKnowledgeException
from mofgbmlpy.fuzzy.knowledge.factory.homo_triangle_knowledge_factory_2_3_4_5 import \
    HomoTriangleKnowledgeFactory_2_3_4_5
from mofgbmlpy.fuzzy.knowledge.factory.homo_triangle_knowledge_factory_5 import HomoTriangleKnowledgeFactory_5
from mofgbmlpy.fuzzy.knowledge.knowledge import Knowledge
from mofgbmlpy.fuzzy.rule.antecedent.factory.heuristic_antecedent_factory import HeuristicAntecedentFactory
from mofgbmlpy.main.arguments import Arguments
from util import get_a0_0_iris_train_test


training_set, _ = get_a0_0_iris_train_test()


def create_example(knowledge=None, is_dc_probability=True, dc_rate=0.5, antecedent_number_do_not_dont_care=1, random_gen=np.random.Generator(np.random.MT19937(seed=2022))):
    if knowledge is None:
        knowledge = HomoTriangleKnowledgeFactory_2_3_4_5(training_set.get_num_dim()).create()
    return HeuristicAntecedentFactory(training_set, knowledge, is_dc_probability, dc_rate, antecedent_number_do_not_dont_care, random_gen), knowledge


def test_none_knowledge():
    random_gen = np.random.Generator(np.random.MT19937(seed=2022))
    knowledge = None
    is_dc_probability = True
    dc_rate = 0.5
    antecedent_number_do_not_dont_care = 1

    with pytest.raises(TypeError):
        HeuristicAntecedentFactory(training_set, knowledge, is_dc_probability, dc_rate, antecedent_number_do_not_dont_care, random_gen)


def test_no_fuzzy_vars_knowledge():
    random_gen = np.random.Generator(np.random.MT19937(seed=2022))
    knowledge = Knowledge()
    is_dc_probability = True
    dc_rate = 0.5
    antecedent_number_do_not_dont_care = 1

    with pytest.raises(UninitializedKnowledgeException):
        HeuristicAntecedentFactory(training_set, knowledge, is_dc_probability, dc_rate, antecedent_number_do_not_dont_care, random_gen)


def test_none_training_set():
    random_gen = np.random.Generator(np.random.MT19937(seed=2022))
    knowledge = HomoTriangleKnowledgeFactory_2_3_4_5(1).create()
    is_dc_probability = True
    dc_rate = 0.5
    antecedent_number_do_not_dont_care = 1
    with pytest.raises(TypeError):
        HeuristicAntecedentFactory(None, knowledge, is_dc_probability, dc_rate, antecedent_number_do_not_dont_care, random_gen)


def test_different_num_dim_training_set():
    random_gen = np.random.Generator(np.random.MT19937(seed=2022))
    knowledge = HomoTriangleKnowledgeFactory_2_3_4_5(1).create()
    is_dc_probability = True
    dc_rate = 0.5
    antecedent_number_do_not_dont_care = 1

    with pytest.raises(ValueError):
        HeuristicAntecedentFactory(training_set, knowledge, is_dc_probability, dc_rate, antecedent_number_do_not_dont_care, random_gen)


def test_is_dc_probability_none():
    random_gen = np.random.Generator(np.random.MT19937(seed=2022))
    knowledge = HomoTriangleKnowledgeFactory_2_3_4_5(training_set.get_num_dim()).create()
    is_dc_probability = None  # will be converted to False
    dc_rate = 0.5
    antecedent_number_do_not_dont_care = 1

    factory = HeuristicAntecedentFactory(training_set, knowledge, is_dc_probability, dc_rate, antecedent_number_do_not_dont_care, random_gen)
    assert not factory.get_is_dc_probability()


@pytest.mark.parametrize("value", np.random.uniform(-1, 1, 5))
def test_dc_rate(value):
    random_gen = np.random.Generator(np.random.MT19937(seed=2022))
    knowledge = HomoTriangleKnowledgeFactory_2_3_4_5(training_set.get_num_dim()).create()
    is_dc_probability = True
    dc_rate = value
    antecedent_number_do_not_dont_care = 1

    if dc_rate < 0 or dc_rate > 1:
        with pytest.raises(ValueError):
            HeuristicAntecedentFactory(training_set, knowledge, is_dc_probability, dc_rate, antecedent_number_do_not_dont_care, random_gen)
    else:
        HeuristicAntecedentFactory(training_set, knowledge, is_dc_probability, dc_rate, antecedent_number_do_not_dont_care, random_gen)


def test_antecedent_number_do_not_dont_care_negative():
    random_gen = np.random.Generator(np.random.MT19937(seed=2022))
    knowledge = HomoTriangleKnowledgeFactory_2_3_4_5(training_set.get_num_dim()).create()
    is_dc_probability = True
    dc_rate = 0.5
    antecedent_number_do_not_dont_care = -1

    with pytest.raises(ValueError):
        HeuristicAntecedentFactory(training_set, knowledge, is_dc_probability, dc_rate, antecedent_number_do_not_dont_care, random_gen)


def test_calculate_antecedent_part_none_pattern():
    factory, _ = create_example()
    pattern = None

    with pytest.raises(TypeError):
        factory.calculate_antecedent_part_py(pattern)


def test_calculate_antecedent_part_empty_pattern():
    factory, _ = create_example()
    pattern = Pattern(0, np.empty(0), ClassLabelBasic(0))

    with pytest.raises(ValueError):
        factory.calculate_antecedent_part_py(pattern)


def test_calculate_antecedent_part_different_dimension_than_knowledge():
    factory, _ = create_example()
    pattern = Pattern(0, np.array([1.0]), ClassLabelBasic(0))

    with pytest.raises(ValueError):
        factory.calculate_antecedent_part_py(pattern)


def test_calculate_antecedent_part_no_dc_from_dc_rate():
    factory, _ = create_example(dc_rate=0)
    pattern = Pattern(0, np.array([0.2, 0.4, 0.6, 0.8]), ClassLabelBasic(0))

    antecedent_indices = factory.calculate_antecedent_part_py(pattern)
    for i in range(len(antecedent_indices)):
        assert antecedent_indices[i] != 0


def test_calculate_antecedent_part_categorical_attribute():
    factory, _ = create_example(dc_rate=0)
    pattern = Pattern(0, np.array([-1.0, -2.0, -3.0, -4.0]), ClassLabelBasic(0))

    antecedent_indices = factory.calculate_antecedent_part_py(pattern)
    for i in range(len(antecedent_indices)):
        assert antecedent_indices[i] == pattern.get_attribute_value(i)


def test_calculate_antecedent_part_all_dc_from_dc_rate():
    factory, _ = create_example(dc_rate=1)
    pattern = Pattern(0, np.array([0.2, 0.4, 0.6, 0.8]), ClassLabelBasic(0))

    antecedent_indices = factory.calculate_antecedent_part_py(pattern)
    for i in range(len(antecedent_indices)):
        assert antecedent_indices[i] == 0


def test_calculate_antecedent_part_all_dc_from_not_dc_probability():
    factory, _ = create_example(antecedent_number_do_not_dont_care=0, is_dc_probability=False)
    pattern = Pattern(0, np.array([0.2, 0.4, 0.6, 0.8]), ClassLabelBasic(0))

    antecedent_indices = factory.calculate_antecedent_part_py(pattern)
    for i in range(len(antecedent_indices)):
        assert antecedent_indices[i] == 0


def test_create_num_rules_negative():
    factory, _ = create_example()

    with pytest.raises(ValueError):
        factory.create_py(-1)


def test_create_num_rules_null():
    factory, _ = create_example()

    with pytest.raises(ValueError):
        factory.create_py(0)


def test_create_3():
    factory, _ = create_example()
    antecedents = factory.create_py(3)

    assert antecedents is not None
    for antecedent in antecedents:
        assert antecedent.get_array_size() == 4
    assert len(antecedents) == 3


def test_create_1():
    factory, _ = create_example()
    antecedents = factory.create_py(1)

    assert antecedents is not None
    for antecedent in antecedents:
        assert antecedent.get_array_size() == 4
    assert len(antecedents) == 1


def test_create_antecedent_indices_from_pattern_none_pattern():
    factory, _ = create_example()
    pattern = None

    with pytest.raises(TypeError):
        factory.create_antecedent_indices_from_pattern_py(pattern)


def test_create_antecedent_indices_from_pattern():
    factory, _ = create_example()
    pattern = training_set.get_pattern(0)

    antecedent_indices = factory.create_antecedent_indices_from_pattern_py(pattern)

    assert antecedent_indices.shape == (1, 4)


def test_create_antecedent_indices_num_rules_negative():
    factory, _ = create_example()

    with pytest.raises(ValueError):
        factory.create_antecedent_indices_py(-1)


def test_create_antecedent_indices_num_rules_null():
    factory, _ = create_example()

    with pytest.raises(ValueError):
        factory.create_antecedent_indices_py(0)


def test_create_antecedent_indices_num_rules_greater_than_dataset():
    custom_training_set = Dataset(4, 4, 3, np.array([
        Pattern(0, np.array([0.5, 0.0, 1.0, 0.0]), ClassLabelBasic(0)),
        Pattern(1, np.array([0.0, 0.5, 0.0, 1.0]), ClassLabelBasic(1)),
        Pattern(2, np.array([1.0, 0.5, 0.0, 0.0]), ClassLabelBasic(2)),
        Pattern(3, np.array([0.5, 1.0, 0.0, 0.0]), ClassLabelBasic(0)),
    ]))

    random_gen = np.random.Generator(np.random.MT19937(seed=2022))
    knowledge = HomoTriangleKnowledgeFactory_2_3_4_5(4).create()
    factory = HeuristicAntecedentFactory(custom_training_set, knowledge, False, 0.5,
                                      5, random_gen)

    antecedent_indices = factory.create_antecedent_indices_py(5)

    assert len(antecedent_indices) == 5


def test_create_antecedent_indices_3():
    factory, _ = create_example()
    antecedents = factory.create_antecedent_indices_py(3)

    assert antecedents.shape == (3, 4)


def test_create_antecedent_indices_1():
    factory, _ = create_example()
    antecedents = factory.create_antecedent_indices_py(1)

    assert antecedents.shape == (1, 4)


def test_eq_different_knowledge():
    knowledge1 = HomoTriangleKnowledgeFactory_2_3_4_5(training_set.get_num_dim()).create()
    knowledge2 = HomoTriangleKnowledgeFactory_5(training_set.get_num_dim()).create()

    random_gen = np.random.Generator(np.random.MT19937(seed=2022))

    factory1, _ = create_example(knowledge=knowledge1, random_gen=random_gen)
    factory2, _ = create_example(knowledge=knowledge2, random_gen=random_gen)

    assert factory1 != factory2


def test_eq_different_is_dc_probability():
    random_gen = np.random.Generator(np.random.MT19937(seed=2022))

    factory1, _ = create_example(random_gen=random_gen, is_dc_probability=True)
    factory2, _ = create_example(random_gen=random_gen, is_dc_probability=False)

    assert factory1 != factory2


def test_eq_different_dc_rate():
    random_gen = np.random.Generator(np.random.MT19937(seed=2022))

    factory1, _ = create_example(random_gen=random_gen, dc_rate=0.5)
    factory2, _ = create_example(random_gen=random_gen, dc_rate=0.6)

    assert factory1 != factory2


def test_eq_different_antecedent_number_do_not_dont_care():
    random_gen = np.random.Generator(np.random.MT19937(seed=2022))

    factory1, _ = create_example(random_gen=random_gen, antecedent_number_do_not_dont_care=5)
    factory2, _ = create_example(random_gen=random_gen, antecedent_number_do_not_dont_care=6)

    assert factory1 != factory2


def test_eq_different_random_gen():
    random_gen1 = np.random.Generator(np.random.MT19937(seed=2022))
    random_gen2 = np.random.Generator(np.random.MT19937(seed=2022))

    factory1, _ = create_example(random_gen=random_gen1)
    factory2, _ = create_example(random_gen=random_gen2)

    assert factory1 != factory2


def test_eq_same():
    random_gen = np.random.Generator(np.random.MT19937(seed=2022))

    factory1, _ = create_example(random_gen=random_gen)
    factory2, _ = create_example(random_gen=random_gen)

    assert factory1 == factory2


def test_deep_copy():
    # Just check if it raises an exception
    obj = create_example(antecedent_number_do_not_dont_care=0, is_dc_probability=False)
    copy_obj = copy.deepcopy(obj)

    assert obj == copy_obj
