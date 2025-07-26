from xml.dom import minidom
import xml.etree.cElementTree as xml_tree

import pytest
import numpy as np
import copy

from mofgbmlpy.fuzzy.fuzzy_term.fuzzy_set.triangular_fuzzy_set import TriangularFuzzySet
from mofgbmlpy.fuzzy.fuzzy_term.fuzzy_variable import FuzzyVariable
from mofgbmlpy.fuzzy.knowledge.factory.homo_triangle_knowledge_factory_2_3_4_5 import (
    HomoTriangleKnowledgeFactory_2_3_4_5,
)
from mofgbmlpy.fuzzy.knowledge.knowledge import Knowledge
from mofgbmlpy.fuzzy.rule.antecedent.antecedent import Antecedent

from mofgbmlpy.fuzzy.fuzzy_term.fuzzy_set.dont_care_fuzzy_set import DontCareFuzzySet
from util import get_a0_0_iris_train_test


def create_test_knowledge():
    fuzzy_vars = np.array(
        [
            FuzzyVariable(fuzzy_sets=np.array([DontCareFuzzySet(0), TriangularFuzzySet(0, 0.3, 0.5, 1, "small")], object)),
            FuzzyVariable(fuzzy_sets=np.array([DontCareFuzzySet(0), TriangularFuzzySet(0, 0.3, 0.5, 1, "small"), TriangularFuzzySet(0.3, 0.7, 1, 2, "medium")], object))
        ], object
    )
    return Knowledge(fuzzy_vars)


def test_none_antecedent():
    knowledge = create_test_knowledge()
    with pytest.raises(Exception):
        Antecedent(None, knowledge)


def test_none_knowledge():
    with pytest.raises(Exception):
        Antecedent(np.empty(0, int), None)


def test_get_array_size_empty():
    knowledge = create_test_knowledge()
    with pytest.raises(Exception):
        antecedent = Antecedent(np.empty(0, int), knowledge)
        assert antecedent.get_array_size() == 0


def test_set_antecedent_indices_none():
    knowledge = create_test_knowledge()
    with pytest.raises(Exception):
        antecedent = Antecedent(np.array([0, 1], int), knowledge)
        antecedent.set_antecedent_indices(None)


def test_get_compatible_grade_no_knowledge():
    with pytest.raises(Exception):
        antecedent = Antecedent(np.array([0, 1], int), Knowledge())
        vector = np.array([1.0, 2.0], np.float64)
        antecedent.get_membership_values(vector)


def test_get_compatible_grade_smaller_num_vars_knowledge():
    knowledge = create_test_knowledge()

    with pytest.raises(Exception):
        antecedent = Antecedent(np.array([0, 0, 0], int), knowledge)
        vector = np.array([1.0, 1.0, 1.0], np.float64)
        antecedent.get_membership_values(vector)


def test_get_compatible_grade_smaller_num_fuzzy_sets_knowledge():
    knowledge = create_test_knowledge()
    with pytest.raises(Exception):
        antecedent = Antecedent(np.array([1], int), knowledge)
        vector = np.array([1.0], np.float64)

        antecedent.get_membership_values(vector)


def test_get_compatible_grade_none_vector():
    knowledge = create_test_knowledge()
    antecedent = Antecedent(np.array([0, 0], int), knowledge)
    vector = None
    with pytest.raises(Exception):
        antecedent.get_membership_values(vector)


def test_get_compatible_grade_too_small_vector():
    knowledge = create_test_knowledge()
    antecedent = Antecedent(np.array([0, 0], int), knowledge)
    vector = np.array([1.0])
    with pytest.raises(Exception):
        antecedent.get_membership_values(vector)


def test_get_compatible_grade_too_big_vector():
    knowledge = create_test_knowledge()

    antecedent = Antecedent(np.array([0, 0], int), knowledge)
    vector = np.array([1.0, 2.0, 1.0])
    with pytest.raises(Exception):
        antecedent.get_membership_values(vector)


def test_get_compatible_grade_invalid_vector_different_sign_1():
    knowledge = create_test_knowledge()
    antecedent = Antecedent(np.array([1, 0], int), knowledge)

    vector = np.array([-2.0, 0], np.float64)
    with pytest.raises(Exception):
        antecedent.get_membership_values(vector)


def test_get_compatible_grade_invalid_vector_different_sign_2():
    knowledge = create_test_knowledge()
    antecedent = Antecedent(np.array([-1, 0], int), knowledge)

    vector = np.array([1.0, 0], np.float64)
    with pytest.raises(Exception):
        antecedent.get_membership_values(vector)


def test_get_compatible_grade_value_smaller_num_vars_knowledge():
    knowledge = create_test_knowledge()

    with pytest.raises(Exception):
        antecedent = Antecedent(np.array([0, 0, 0], int), knowledge)
        vector = np.array([1.0, 1.0, 1.0])
        antecedent.get_compatible_grade_value_py(vector)


def test_get_compatible_grade_value_smaller_num_fuzzy_sets_knowledge():
    knowledge = create_test_knowledge()
    with pytest.raises(Exception):
        antecedent = Antecedent(np.array([1], int), knowledge)
        vector = np.array([1.0], np.float64)

        antecedent.get_compatible_grade_value_py(vector)


def test_get_compatible_grade_value_none_vector():
    knowledge = create_test_knowledge()
    antecedent = Antecedent(np.array([0, 0], int), knowledge)
    vector = None

    with pytest.raises(Exception):
        antecedent.get_compatible_grade_value_py(vector)


def test_get_compatible_grade_value_too_small_vector():
    knowledge = create_test_knowledge()
    antecedent = Antecedent(np.array([0, 0], int), knowledge)
    vector = np.array([1.0])
    with pytest.raises(Exception):
        antecedent.get_compatible_grade_value_py(vector)


def test_get_compatible_grade_value_too_big_vector():
    knowledge = create_test_knowledge()

    antecedent = Antecedent(np.array([0, 0], int), knowledge)
    vector = np.array([1.0, 2.0, 1.0])
    with pytest.raises(Exception):
        antecedent.get_compatible_grade_value_py(vector)


def test_get_compatible_grade_value_invalid_vector_different_sign_1():
    knowledge = create_test_knowledge()
    antecedent = Antecedent(np.array([1, 0], int), knowledge)

    vector = np.array([-2.0, 0], np.float64)
    with pytest.raises(Exception):
        antecedent.get_compatible_grade_value_py(vector)


def test_get_compatible_grade_value_invalid_vector_different_sign_2():
    knowledge = create_test_knowledge()
    antecedent = Antecedent(np.array([-1, 0], int), knowledge)

    vector = np.array([1.0, 0], np.float64)
    with pytest.raises(Exception):
        antecedent.get_compatible_grade_value_py(vector)


def test_get_length_empty():
    with pytest.raises(Exception):
        _ = Antecedent(np.empty(0, int), Knowledge())


def test_get_length():
    knowledge = create_test_knowledge()
    antecedent = Antecedent(np.array([0, 1], int), knowledge)
    assert antecedent.get_length() == 1


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
    knowledge1.set_fuzzy_vars(np.array([FuzzyVariable(fuzzy_sets=np.array(
        [DontCareFuzzySet(0), TriangularFuzzySet(0, 0.3, 0.5, 1, "small")], object))], object))

    antecedent1 = Antecedent(np.array([1]), knowledge1)
    antecedent2 = Antecedent(np.array([0, 1, 2]), knowledge2)

    assert antecedent1 != antecedent2


def test_eq_different_size_antecedent():
    knowledge1 = HomoTriangleKnowledgeFactory_2_3_4_5(2).create()
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

    assert antecedent == antecedent_copy and id(antecedent.get_antecedent_indices().base) != id(
        antecedent_copy.get_antecedent_indices().base
    )


def test_get_compatible_grade_value_example():
    train, _ = get_a0_0_iris_train_test()
    pattern = train.get_pattern(30)

    knowledge = HomoTriangleKnowledgeFactory_2_3_4_5(train.get_num_dim()).create()

    antecedent = Antecedent(np.array([6, 0, 10, 0], dtype=np.int32), knowledge)
    membership_values = antecedent.get_membership_values(pattern.get_attributes_vector())
    compatible_grade_value = antecedent.get_compatible_grade_value_py(pattern.get_attributes_vector())

    assert 0 == membership_values[0]
    assert 1 == membership_values[1]
    assert pytest.approx(0.7288135290145874, rel=1e-6) == membership_values[2]
    assert 1 == membership_values[3]

    assert compatible_grade_value == 0
