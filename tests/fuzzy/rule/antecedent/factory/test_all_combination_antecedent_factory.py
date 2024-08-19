import copy

import numpy as np
import pytest

from mofgbmlpy.exception.uninitialized_knowledge_exception import UninitializedKnowledgeException
from mofgbmlpy.fuzzy.knowledge.factory.homo_triangle_knowledge_factory_2_3_4_5 import \
    HomoTriangleKnowledgeFactory_2_3_4_5
from mofgbmlpy.fuzzy.knowledge.knowledge import Knowledge
from mofgbmlpy.fuzzy.rule.antecedent.antecedent import Antecedent
from mofgbmlpy.fuzzy.rule.antecedent.factory.all_combination_antecedent_factory import AllCombinationAntecedentFactory


def test_none_knowledge():
    knowledge = None
    random_gen = np.random.Generator(np.random.MT19937(seed=2022))
    with pytest.raises(TypeError):
        AllCombinationAntecedentFactory(knowledge, random_gen)


def test_no_fuzzy_vars_knowledge():
    knowledge = Knowledge()
    random_gen = np.random.Generator(np.random.MT19937(seed=2022))
    with pytest.raises(UninitializedKnowledgeException):
        AllCombinationAntecedentFactory(knowledge, random_gen)


@pytest.mark.parametrize("num_dim", np.random.randint(1, 5, 3))
def test_generate_antecedents_indices_check_number_results(num_dim):
    knowledge = HomoTriangleKnowledgeFactory_2_3_4_5(num_dim).create()
    random_gen = np.random.Generator(np.random.MT19937(seed=2022))
    expected = 1
    for i in range(knowledge.get_num_dim()):
        expected *= knowledge.get_num_fuzzy_sets(i)
    factory = AllCombinationAntecedentFactory(knowledge, random_gen)
    assert expected == factory.get_num_antecedents()


def test_create_negative_num_rules():
    random_gen = np.random.Generator(np.random.MT19937(seed=2022))
    knowledge = HomoTriangleKnowledgeFactory_2_3_4_5(3).create()
    factory = AllCombinationAntecedentFactory(knowledge, random_gen)

    with pytest.raises(ValueError):
        factory.create_py(-1)


def test_create_null_num_rules():
    random_gen = np.random.Generator(np.random.MT19937(seed=2022))
    knowledge = HomoTriangleKnowledgeFactory_2_3_4_5(3).create()
    factory = AllCombinationAntecedentFactory(knowledge, random_gen)

    with pytest.raises(ValueError):
        factory.create_py(0)


def test_create_1_rule():
    knowledge = HomoTriangleKnowledgeFactory_2_3_4_5(3).create()
    random_gen = np.random.Generator(np.random.MT19937(seed=2022))
    factory = AllCombinationAntecedentFactory(knowledge, random_gen)
    antecedent = factory.create_py(1)

    assert len(antecedent) == 1 and isinstance(antecedent[0], Antecedent)


def test_create_5_rule():
    knowledge = HomoTriangleKnowledgeFactory_2_3_4_5(3).create()
    random_gen = np.random.Generator(np.random.MT19937(seed=2022))
    factory = AllCombinationAntecedentFactory(knowledge, random_gen)
    antecedent = factory.create_py(5)

    assert len(antecedent) == 5 and isinstance(antecedent[0], Antecedent)


def test_create_antecedent_indices_negative_num_rules():
    knowledge = HomoTriangleKnowledgeFactory_2_3_4_5(3).create()
    random_gen = np.random.Generator(np.random.MT19937(seed=2022))
    factory = AllCombinationAntecedentFactory(knowledge, random_gen)

    with pytest.raises(ValueError):
        factory.create_antecedent_indices_py(-1)


def test_create_antecedent_indices_null_num_rules():
    knowledge = HomoTriangleKnowledgeFactory_2_3_4_5(3).create()
    random_gen = np.random.Generator(np.random.MT19937(seed=2022))
    factory = AllCombinationAntecedentFactory(knowledge, random_gen)

    with pytest.raises(ValueError):
        factory.create_antecedent_indices_py(0)


def test_create_antecedent_indices_1_rule():
    knowledge = HomoTriangleKnowledgeFactory_2_3_4_5(3).create()
    random_gen = np.random.Generator(np.random.MT19937(seed=2022))
    factory = AllCombinationAntecedentFactory(knowledge, random_gen)
    antecedent = factory.create_antecedent_indices_py(1)

    assert len(antecedent) == 1 and len(antecedent[0]) == 3 and antecedent[0][0] >= 0


def test_create_antecedent_indices_5_rule():
    knowledge = HomoTriangleKnowledgeFactory_2_3_4_5(3).create()
    random_gen = np.random.Generator(np.random.MT19937(seed=2022))
    factory = AllCombinationAntecedentFactory(knowledge, random_gen)
    antecedent = factory.create_antecedent_indices_py(5)

    assert len(antecedent) == 5 and len(antecedent[0]) == 3 and antecedent[0][0] >= 0


def test_eq_true():
    knowledge1 = HomoTriangleKnowledgeFactory_2_3_4_5(3).create()
    knowledge2 = HomoTriangleKnowledgeFactory_2_3_4_5(3).create()
    random_gen = np.random.Generator(np.random.MT19937(seed=2022))
    factory1 = AllCombinationAntecedentFactory(knowledge1, random_gen)
    factory2 = AllCombinationAntecedentFactory(knowledge2, random_gen)

    assert factory1 == factory2


def test_eq_false():
    knowledge1 = HomoTriangleKnowledgeFactory_2_3_4_5(3).create()
    knowledge2 = HomoTriangleKnowledgeFactory_2_3_4_5(4).create()
    random_gen = np.random.Generator(np.random.MT19937(seed=2022))
    factory1 = AllCombinationAntecedentFactory(knowledge1, random_gen)
    factory2 = AllCombinationAntecedentFactory(knowledge2, random_gen)

    assert factory1 != factory2


def test_deep_copy():
    # Just check if it raises an exception
    random_gen = np.random.Generator(np.random.MT19937(seed=2022))
    obj = AllCombinationAntecedentFactory(HomoTriangleKnowledgeFactory_2_3_4_5(3).create(), random_gen)
    _ = copy.deepcopy(obj)

    assert True