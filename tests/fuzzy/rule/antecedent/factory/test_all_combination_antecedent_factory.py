import numpy as np
import pytest

from mofgbmlpy.fuzzy.knowledge.factory.homo_triangle_knowledge_factory_2_3_4_5 import \
    HomoTriangleKnowledgeFactory_2_3_4_5
from mofgbmlpy.fuzzy.knowledge.knowledge import Knowledge
from mofgbmlpy.fuzzy.rule.antecedent.antecedent import Antecedent
from mofgbmlpy.fuzzy.rule.antecedent.factory.all_combination_antecedent_factory import AllCombinationAntecedentFactory


def test_none_knowledge():
    knowledge = None
    with pytest.raises(Exception):
        AllCombinationAntecedentFactory(knowledge)


def test_no_fuzzy_vars_knowledge():
    knowledge = Knowledge()
    with pytest.raises(Exception):
        AllCombinationAntecedentFactory(knowledge)


@pytest.mark.parametrize("num_dim", np.random.randint(1, 5, 3))
def test_generate_antecedents_indices_check_number_results(num_dim):
    knowledge = HomoTriangleKnowledgeFactory_2_3_4_5(num_dim).create()

    expected = 1
    for i in range(knowledge.get_num_dim()):
        expected *= knowledge.get_num_fuzzy_sets(i)
    factory = AllCombinationAntecedentFactory(knowledge)
    assert expected == factory.get_num_antecedents()


def test_create_negative_num_rules():
    knowledge = HomoTriangleKnowledgeFactory_2_3_4_5(3).create()
    factory = AllCombinationAntecedentFactory(knowledge)

    with pytest.raises(Exception):
        factory.create(-1)


def test_create_null_num_rules():
    knowledge = HomoTriangleKnowledgeFactory_2_3_4_5(3).create()
    factory = AllCombinationAntecedentFactory(knowledge)

    with pytest.raises(Exception):
        factory.create(0)


def test_create_1_rule():
    knowledge = HomoTriangleKnowledgeFactory_2_3_4_5(3).create()
    factory = AllCombinationAntecedentFactory(knowledge)
    antecedent = factory.create_py(1)

    assert len(antecedent) == 1 and isinstance(antecedent[0], Antecedent)


def test_create_5_rule():
    knowledge = HomoTriangleKnowledgeFactory_2_3_4_5(3).create()
    factory = AllCombinationAntecedentFactory(knowledge)
    antecedent = factory.create_py(5)

    assert len(antecedent) == 5 and isinstance(antecedent[0], Antecedent)


def test_create_antecedent_indices_negative_num_rules():
    knowledge = HomoTriangleKnowledgeFactory_2_3_4_5(3).create()
    factory = AllCombinationAntecedentFactory(knowledge)

    with pytest.raises(Exception):
        factory.create_antecedent_indices_py(-1)


def test_create_antecedent_indices_null_num_rules():
    knowledge = HomoTriangleKnowledgeFactory_2_3_4_5(3).create()
    factory = AllCombinationAntecedentFactory(knowledge)

    with pytest.raises(Exception):
        factory.create_antecedent_indices_py(0)


def test_create_antecedent_indices_1_rule():
    knowledge = HomoTriangleKnowledgeFactory_2_3_4_5(3).create()
    factory = AllCombinationAntecedentFactory(knowledge)
    antecedent = factory.create_antecedent_indices_py(1)

    assert len(antecedent) == 1 and len(antecedent[0]) == 3 and antecedent[0][0] >= 0


def test_create_antecedent_indices_5_rule():
    knowledge = HomoTriangleKnowledgeFactory_2_3_4_5(3).create()
    factory = AllCombinationAntecedentFactory(knowledge)
    antecedent = factory.create_antecedent_indices_py(5)

    assert len(antecedent) == 5 and len(antecedent[0]) == 3 and antecedent[0][0] >= 0


def test_eq_true():
    knowledge1 = HomoTriangleKnowledgeFactory_2_3_4_5(3).create()
    knowledge2 = HomoTriangleKnowledgeFactory_2_3_4_5(3).create()
    factory1 = AllCombinationAntecedentFactory(knowledge1)
    factory2 = AllCombinationAntecedentFactory(knowledge2)

    assert factory1 == factory2


def test_eq_false():
    knowledge1 = HomoTriangleKnowledgeFactory_2_3_4_5(3).create()
    knowledge2 = HomoTriangleKnowledgeFactory_2_3_4_5(4).create()
    factory1 = AllCombinationAntecedentFactory(knowledge1)
    factory2 = AllCombinationAntecedentFactory(knowledge2)

    assert factory1 != factory2
