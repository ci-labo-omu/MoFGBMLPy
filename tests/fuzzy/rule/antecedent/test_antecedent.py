from xml.dom import minidom
import xml.etree.cElementTree as xml_tree

import pytest
import numpy as np
import copy

from mofgbmlpy.fuzzy.fuzzy_term.fuzzy_set.triangular_fuzzy_set import TriangularFuzzySet
from mofgbmlpy.fuzzy.fuzzy_term.fuzzy_variable import FuzzyVariable
from mofgbmlpy.fuzzy.knowledge.factory.homo_triangle_knowledge_factory_2_3_4_5 import \
    HomoTriangleKnowledgeFactory_2_3_4_5
from mofgbmlpy.fuzzy.knowledge.knowledge import Knowledge
from mofgbmlpy.fuzzy.rule.antecedent.antecedent import Antecedent


def test_none_antecedent():
    with pytest.raises(Exception):
        Antecedent(None, Knowledge())


def test_none_knowledge():
    with pytest.raises(Exception):
        Antecedent(np.empty(0, int), None)


def test_get_array_size_empty():
    antecedent = Antecedent(np.empty(0, int), Knowledge())
    assert antecedent.get_array_size() == 0


def test_set_antecedent_indices_none():
    antecedent = Antecedent(np.array([0, 1], int), Knowledge())
    with pytest.raises(Exception):
        antecedent.set_antecedent_indices(None)


def test_get_compatible_grade_no_knowledge():
    antecedent = Antecedent(np.array([0, 1], int), Knowledge())
    vector = np.array([1.0, 2.0])
    with pytest.raises(Exception):
        antecedent.get_compatible_grade(vector)


def test_get_compatible_grade_smaller_num_vars_knowledge():
    fuzzy_vars = np.array([FuzzyVariable(fuzzy_sets=np.array([TriangularFuzzySet(0, 0.5, 1, 0, "small")], object))], object)

    antecedent = Antecedent(np.array([0, 0], int), Knowledge(fuzzy_vars))
    vector = np.array([1.0, 2.0])
    with pytest.raises(Exception):
        antecedent.get_compatible_grade(vector)


def test_get_compatible_grade_smaller_num_fuzzy_sets_knowledge():
    fuzzy_vars = np.array([FuzzyVariable(fuzzy_sets=np.array([TriangularFuzzySet(0, 0.5, 1, 0, "small")], object))], object)

    knowledge = Knowledge(fuzzy_vars)
    antecedent = Antecedent(np.array([1], int), knowledge)
    vector = np.array([1.0])

    with pytest.raises(Exception):
        antecedent.get_compatible_grade(vector)


def test_get_compatible_grade_none_vector():
    antecedent = Antecedent(np.array([0, 1], int), Knowledge())
    vector = None
    with pytest.raises(Exception):
        antecedent.get_compatible_grade(vector)


def test_get_compatible_grade_too_small_vector():
    antecedent = Antecedent(np.array([0, 1], int), Knowledge())
    vector = np.array([1.0])
    with pytest.raises(Exception):
        antecedent.get_compatible_grade(vector)


def test_get_compatible_grade_too_big_vector():
    antecedent = Antecedent(np.array([0, 1], int), Knowledge())
    vector = np.array([1.0, 2.0, 1.0])
    with pytest.raises(Exception):
        antecedent.get_compatible_grade(vector)


def test_get_compatible_grade_invalid_vector_different_sign_1():
    antecedent = Antecedent(np.array([0, 1], int), Knowledge())
    vector = np.array([1.0, -2.0])
    with pytest.raises(Exception):
        antecedent.get_compatible_grade(vector)


def test_get_compatible_grade_invalid_vector_different_sign_2():
    antecedent = Antecedent(np.array([1, -2], int), Knowledge())
    vector = np.array([1.0, 2.0])
    with pytest.raises(Exception):
        antecedent.get_compatible_grade(vector)


def test_get_compatible_grade_value_no_knowledge():
    antecedent = Antecedent(np.array([0, 1], int), Knowledge())
    vector = np.array([1.0, 2.0])
    with pytest.raises(Exception):
        antecedent.get_compatible_grade_value_py(vector)


def test_get_compatible_grade_value_smaller_num_vars_knowledge():
    fuzzy_vars = np.array([FuzzyVariable(fuzzy_sets=np.array([TriangularFuzzySet(0, 0.5, 1, 0, "small")], object))], object)

    antecedent = Antecedent(np.array([0, 0], int), Knowledge(fuzzy_vars))
    vector = np.array([1.0, 2.0])
    with pytest.raises(Exception):
        antecedent.get_compatible_grade_value_py(vector)


def test_get_compatible_grade_value_smaller_num_fuzzy_sets_knowledge():
    fuzzy_vars = np.array([FuzzyVariable(fuzzy_sets=np.array([TriangularFuzzySet(0, 0.5, 1, 0, "small")], object))], object)

    knowledge = Knowledge(fuzzy_vars)
    antecedent = Antecedent(np.array([1], int), knowledge)
    vector = np.array([1.0])

    with pytest.raises(Exception):
        antecedent.get_compatible_grade_value_py(vector)


def test_get_compatible_grade_value_none_vector():
    antecedent = Antecedent(np.array([0, 1], int), Knowledge())
    vector = None
    with pytest.raises(Exception):
        antecedent.get_compatible_grade_value_py(vector)


def test_get_compatible_grade_value_too_small_vector():
    antecedent = Antecedent(np.array([0, 1], int), Knowledge())
    vector = np.array([1.0])
    with pytest.raises(Exception):
        antecedent.get_compatible_grade_value_py(vector)


def test_get_compatible_grade_value_too_big_vector():
    antecedent = Antecedent(np.array([0, 1], int), Knowledge())
    vector = np.array([1.0, 2.0, 1.0])
    with pytest.raises(Exception):
        antecedent.get_compatible_grade_value_py(vector)


def test_get_compatible_grade_value_invalid_vector_different_sign_1():
    antecedent = Antecedent(np.array([0, 1], int), Knowledge())
    vector = np.array([1.0, -2.0])
    with pytest.raises(Exception):
        antecedent.get_compatible_grade_value_py(vector)


def test_get_compatible_grade_value_invalid_vector_different_sign_2():
    antecedent = Antecedent(np.array([1, -2], int), Knowledge())
    vector = np.array([1.0, 2.0])
    with pytest.raises(Exception):
        antecedent.get_compatible_grade_value_py(vector)


def test_get_length_empty():
    antecedent = Antecedent(np.empty(0, int), Knowledge())
    assert antecedent.get_length() == 0


def test_get_length():
    antecedent = Antecedent(np.array([0, 1, -2, 0, 0, 5], int), Knowledge())
    assert antecedent.get_length() == 3


def test_to_xml_run():
    # Only test if it doesn't return an exception
    knowledge = HomoTriangleKnowledgeFactory_2_3_4_5(3).create()
    antecedent = Antecedent(np.array([0, 1, 2]), knowledge)

    reparsed = minidom.parseString(xml_tree.tostring(antecedent.to_xml()))
    _ = reparsed.toprettyxml(indent="  ")

    assert True


def test_eq_true():
    knowledge1 = HomoTriangleKnowledgeFactory_2_3_4_5(3).create()
    knowledge2 = HomoTriangleKnowledgeFactory_2_3_4_5(3).create()
    antecedent1 = Antecedent(np.array([0, 1, 2]), knowledge1)
    antecedent2 = Antecedent(np.array([0, 1, 2]), knowledge2)

    assert antecedent1 == antecedent2


def test_eq_different_knowledge():
    knowledge1 = HomoTriangleKnowledgeFactory_2_3_4_5(4).create()
    knowledge2 = HomoTriangleKnowledgeFactory_2_3_4_5(3).create()
    antecedent1 = Antecedent(np.array([0, 1, 2]), knowledge1)
    antecedent2 = Antecedent(np.array([0, 1, 2]), knowledge2)

    assert antecedent1 != antecedent2


def test_eq_different_size_antecedent():
    knowledge1 = HomoTriangleKnowledgeFactory_2_3_4_5(3).create()
    knowledge2 = HomoTriangleKnowledgeFactory_2_3_4_5(3).create()
    antecedent1 = Antecedent(np.array([0, 1]), knowledge1)
    antecedent2 = Antecedent(np.array([0, 1, 2]), knowledge2)

    assert antecedent1 != antecedent2


def test_eq_different_order_antecedent():
    knowledge1 = HomoTriangleKnowledgeFactory_2_3_4_5(3).create()
    knowledge2 = HomoTriangleKnowledgeFactory_2_3_4_5(3).create()
    antecedent1 = Antecedent(np.array([0, 2, 1]), knowledge1)
    antecedent2 = Antecedent(np.array([0, 1, 2]), knowledge2)

    assert antecedent1 != antecedent2


def test_deepcopy():
    knowledge = HomoTriangleKnowledgeFactory_2_3_4_5(3).create()
    antecedent = Antecedent(np.array([0, 2, 1]), knowledge)
    antecedent_copy = copy.deepcopy(antecedent)

    assert (antecedent == antecedent_copy and
            id(antecedent.get_antecedent_indices().base) != id(antecedent_copy.get_antecedent_indices().base))
