import copy

import numpy as np
import pytest

from mofgbmlpy.exception.uninitialized_knowledge_exception import UninitializedKnowledgeException
from mofgbmlpy.fuzzy.fuzzy_term.fuzzy_set.dont_care_fuzzy_set import DontCareFuzzySet
from mofgbmlpy.fuzzy.fuzzy_term.fuzzy_set.triangular_fuzzy_set import TriangularFuzzySet
from mofgbmlpy.fuzzy.fuzzy_term.fuzzy_variable import FuzzyVariable
from mofgbmlpy.fuzzy.knowledge.factory.homo_triangle_knowledge_factory_2_3_4_5 import \
    HomoTriangleKnowledgeFactory_2_3_4_5
from mofgbmlpy.fuzzy.knowledge.factory.homo_triangle_knowledge_factory_5 import HomoTriangleKnowledgeFactory_5
from mofgbmlpy.fuzzy.knowledge.knowledge import Knowledge


def test_none_fuzzy_sets():
    var = Knowledge(None)
    assert len(var.get_fuzzy_vars()) == 0


def test_get_fuzzy_variable_out_of_bounds_index_negative():
    fuzzy_vars = np.array([FuzzyVariable(fuzzy_sets=np.array([TriangularFuzzySet(0, 0.5, 1, 0, "small")], object))], object)
    knowledge = Knowledge(fuzzy_vars)
    with pytest.raises(IndexError):
        knowledge.get_fuzzy_variable(-2)


def test_get_fuzzy_variable_out_of_bounds_index_big():
    fuzzy_vars = np.array([FuzzyVariable(fuzzy_sets=np.array([TriangularFuzzySet(0, 0.5, 1, 0, "small")], object))], object)
    knowledge = Knowledge(fuzzy_vars)
    with pytest.raises(IndexError):
        knowledge.get_fuzzy_variable(1)


def test_get_fuzzy_variable_empty_fuzzy_sets_array():
    knowledge = Knowledge()
    with pytest.raises(IndexError):
        knowledge.get_fuzzy_variable(0)


def test_get_fuzzy_set_out_of_bounds_index_negative():
    fuzzy_vars = np.array([FuzzyVariable(fuzzy_sets=np.array([TriangularFuzzySet(0, 0.5, 1, 0, "small")], object))], object)
    knowledge = Knowledge(fuzzy_vars)
    with pytest.raises(IndexError):
        knowledge.get_fuzzy_set(0, -2)


def test_get_fuzzy_set_out_of_bounds_index_big():
    fuzzy_vars = np.array([FuzzyVariable(fuzzy_sets=np.array([TriangularFuzzySet(0, 0.5, 1, 0, "small")], object))], object)
    knowledge = Knowledge(fuzzy_vars)
    with pytest.raises(IndexError):
        knowledge.get_fuzzy_set(0, 1)


def test_get_fuzzy_set_empty_fuzzy_vars_array():
    knowledge = Knowledge()
    with pytest.raises(UninitializedKnowledgeException):
        knowledge.get_fuzzy_set(0, 0)


def test_get_fuzzy_set_out_of_bounds_dim_negative():
    fuzzy_vars = np.array([FuzzyVariable(fuzzy_sets=np.array([TriangularFuzzySet(0, 0.5, 1, 0, "small")], object))], object)
    knowledge = Knowledge(fuzzy_vars)
    with pytest.raises(IndexError):
        knowledge.get_fuzzy_set(-2, 0)


def test_get_fuzzy_set_out_of_bounds_dim_big():
    fuzzy_vars = np.array([FuzzyVariable(fuzzy_sets=np.array([TriangularFuzzySet(0, 0.5, 1, 0, "small")], object))], object)
    knowledge = Knowledge(fuzzy_vars)
    with pytest.raises(IndexError):
        knowledge.get_fuzzy_set(1, 0)


def test_get_num_fuzzy_sets_out_of_bounds_index_negative():
    fuzzy_vars = np.array([FuzzyVariable(fuzzy_sets=np.array([TriangularFuzzySet(0, 0.5, 1, 0, "small")], object))], object)
    knowledge = Knowledge(fuzzy_vars)
    with pytest.raises(IndexError):
        knowledge.get_num_fuzzy_sets(-2)


def test_get_num_fuzzy_sets_out_of_bounds_index_big():
    fuzzy_vars = np.array([FuzzyVariable(fuzzy_sets=np.array([TriangularFuzzySet(0, 0.5, 1, 0, "small")], object))], object)
    knowledge = Knowledge(fuzzy_vars)
    with pytest.raises(IndexError):
        knowledge.get_num_fuzzy_sets(1)


def test_get_num_fuzzy_sets_empty_fuzzy_vars_array():
    knowledge = Knowledge()
    with pytest.raises(UninitializedKnowledgeException):
        knowledge.get_num_fuzzy_sets(0)


@pytest.mark.parametrize("size", np.random.randint(1, 10, size=5))
def test_get_num_fuzzy_sets_valid(size):
    fuzzy_sets = np.empty(size, object)
    for i in range(size):
        fuzzy_sets[i] = TriangularFuzzySet(0, 0.5, 1, 0, "small")
    fuzzy_vars = np.array([FuzzyVariable(fuzzy_sets=fuzzy_sets)], object)
    knowledge = Knowledge(fuzzy_vars)

    assert knowledge.get_num_fuzzy_sets(0) == size


def test_set_fuzzy_vars_none():
    fuzzy_vars = np.array([FuzzyVariable(fuzzy_sets=np.array([TriangularFuzzySet(0, 0.5, 1, 0, "small")], object))], object)
    knowledge = Knowledge(fuzzy_vars)

    knowledge.set_fuzzy_vars(None)
    assert len(knowledge.get_fuzzy_vars()) == 0


def test_set_fuzzy_vars_one():
    fuzzy_vars1 = np.array([FuzzyVariable(fuzzy_sets=np.array([TriangularFuzzySet(0, 0.5, 1, 0, "small")], object))], object)
    knowledge = Knowledge(fuzzy_vars1)
    fuzzy_sets = np.array([TriangularFuzzySet(0, 0.5, 1, 2024, "small"),
                                                TriangularFuzzySet(0, 0.5, 1, 0, "small")],
                                                object)
    fuzzy_vars = np.array([FuzzyVariable(fuzzy_sets=fuzzy_sets)], object)

    knowledge.set_fuzzy_vars(fuzzy_vars)

    assert (len(knowledge.get_fuzzy_vars()) == 1 and
            knowledge.get_num_fuzzy_sets(0) == 2 and
            knowledge.get_fuzzy_set(0, 0).get_id() == 2024)


def test_get_membership_value_empty_vars_array():
    knowledge = Knowledge()
    with pytest.raises(UninitializedKnowledgeException):
        knowledge.get_membership_value_py(0, 0, 0)


def test_get_membership_value_out_of_bounds_dim_negative():
    var1 = FuzzyVariable(fuzzy_sets=np.array([TriangularFuzzySet(0, 0.5, 1, 0, "small")], object))
    var2 = FuzzyVariable(fuzzy_sets=np.array([TriangularFuzzySet(0.1, 0.5, 1.1, 1, "small")], object))

    knowledge = Knowledge(np.array([var1, var2], object))
    with pytest.raises(IndexError):
        knowledge.get_membership_value_py(0, -1, 0)


def test_get_membership_value_out_of_bounds_dim_big():
    var1 = FuzzyVariable(fuzzy_sets=np.array([TriangularFuzzySet(0, 0.5, 1, 0, "small")], object))
    var2 = FuzzyVariable(fuzzy_sets=np.array([TriangularFuzzySet(0.1, 0.5, 1.1, 1, "small")], object))

    knowledge = Knowledge(np.array([var1, var2], object))
    with pytest.raises(IndexError):
        knowledge.get_membership_value_py(0, 2, 0)


def test_get_membership_value_out_of_fuzzy_set_index_negative():
    var = FuzzyVariable(fuzzy_sets=np.array([TriangularFuzzySet(0, 0.5, 1, 0, "small")], object))

    knowledge = Knowledge(np.array([var], object))
    with pytest.raises(IndexError):
        knowledge.get_membership_value_py(0, 0, -2)


def test_get_membership_value_out_of_fuzzy_set_index_big():
    var = FuzzyVariable(fuzzy_sets=np.array([TriangularFuzzySet(0, 0.5, 1, 0, "small")], object))

    knowledge = Knowledge(np.array([var], object))
    with pytest.raises(IndexError):
        knowledge.get_membership_value_py(0, 0, 1)


def test_get_membership_value_valid():
    fuzzy_vars = np.array([FuzzyVariable(fuzzy_sets=np.array([TriangularFuzzySet(0, 0.5, 1, 0, "small")], object))], object)
    knowledge = Knowledge(fuzzy_vars)
    left_val = knowledge.get_membership_value_py(0, 0, 0)
    center_val = knowledge.get_membership_value_py(0.5, 0, 0)
    right_val = knowledge.get_membership_value_py(1, 0, 0)

    assert left_val == 0 and center_val == 1 and right_val == 0


def test_get_num_dims_empty():
    knowledge = Knowledge()
    assert knowledge.get_num_dim() == 0


@pytest.mark.parametrize("num_dims", np.concatenate([[0], np.random.randint(1, 10, size=5)]))
def test_get_num_dim_valid(num_dims):
    fuzzy_vars = np.empty(num_dims, object)
    for i in range(num_dims):
        fuzzy_sets = np.array([TriangularFuzzySet(0, 0.5, 1, 0, "small")], object)
        fuzzy_vars[i] = FuzzyVariable(fuzzy_sets=fuzzy_sets)
    knowledge = Knowledge(fuzzy_vars)

    assert knowledge.get_num_dim() == num_dims


def test_get_support_out_of_bounds_index_negative():
    fuzzy_vars = np.array([FuzzyVariable(fuzzy_sets=np.array([TriangularFuzzySet(0, 0.5, 1, 0, "small")], object))], object)
    knowledge = Knowledge(fuzzy_vars)
    with pytest.raises(IndexError):
        knowledge.get_support(0, -2)


def test_get_support_out_of_bounds_index_big():
    fuzzy_vars = np.array([FuzzyVariable(fuzzy_sets=np.array([TriangularFuzzySet(0, 0.5, 1, 0, "small")], object))], object)
    knowledge = Knowledge(fuzzy_vars)
    with pytest.raises(IndexError):
        knowledge.get_support(0, 1)


def test_get_support_empty_fuzzy_vars_array():
    knowledge = Knowledge()
    with pytest.raises(IndexError):
        knowledge.get_support(0, 0)


def test_get_support_out_of_bounds_dim_negative():
    fuzzy_vars = np.array([FuzzyVariable(fuzzy_sets=np.array([TriangularFuzzySet(0, 0.5, 1, 0, "small")], object))], object)
    knowledge = Knowledge(fuzzy_vars)
    with pytest.raises(IndexError):
        knowledge.get_support(-2, 0)


def test_get_support_out_of_bounds_dim_big():
    fuzzy_vars = np.array([FuzzyVariable(fuzzy_sets=np.array([TriangularFuzzySet(0, 0.5, 1, 0, "small")], object))], object)
    knowledge = Knowledge(fuzzy_vars)
    with pytest.raises(IndexError):
        knowledge.get_support(1, 0)


def test_eq_true():
    fuzzy_vars1 = np.array([FuzzyVariable(fuzzy_sets=np.array([TriangularFuzzySet(0, 0.5, 1, 0, "small")], object))], object)
    knowledge1 = Knowledge(fuzzy_vars1)

    fuzzy_vars2 = np.array([FuzzyVariable(fuzzy_sets=np.array([TriangularFuzzySet(0, 0.5, 1, 0, "small")], object))], object)
    knowledge2 = Knowledge(fuzzy_vars2)

    assert knowledge1 == knowledge2


def test_eq_different_empty():
    fuzzy_vars1 = np.array([FuzzyVariable(fuzzy_sets=np.array([TriangularFuzzySet(0, 0.5, 1, 0, "small")], object))], object)
    knowledge1 = Knowledge(fuzzy_vars1)
    knowledge2 = Knowledge()

    assert knowledge1 != knowledge2


def test_eq_different_size():
    fuzzy_vars1 = np.array([FuzzyVariable(fuzzy_sets=np.array([TriangularFuzzySet(0, 0.5, 1, 0, "small")], object))], object)
    knowledge1 = Knowledge(fuzzy_vars1)

    fuzzy_vars2 = np.array([FuzzyVariable(fuzzy_sets=np.array([TriangularFuzzySet(0, 0.5, 1, 0, "small")], object)),
                            FuzzyVariable(fuzzy_sets=np.array([TriangularFuzzySet(0, 0.5, 1, 0, "small")], object))
                            ], object)
    knowledge2 = Knowledge(fuzzy_vars2)

    assert knowledge1 != knowledge2


def test_eq_different_order():
    var1 = FuzzyVariable(fuzzy_sets=np.array([TriangularFuzzySet(0, 0.5, 1, 0, "small")], object))
    var2 = FuzzyVariable(fuzzy_sets=np.array([TriangularFuzzySet(0.1, 0.5, 1.1, 1, "small")], object))

    knowledge1 = Knowledge(np.array([var1, var2], object))
    knowledge2 = Knowledge(np.array([var2, var1], object))

    assert knowledge1 != knowledge2


def test_deepcopy():
    var1 = FuzzyVariable(fuzzy_sets=np.array([TriangularFuzzySet(0, 0.5, 1, 0, "small")], object))
    var2 = FuzzyVariable(fuzzy_sets=np.array([TriangularFuzzySet(0.1, 0.5, 1.1, 1, "small")], object))

    knowledge = Knowledge(np.array([var1, var2], object))
    knowledge_copy = copy.deepcopy(knowledge)

    assert (knowledge == knowledge_copy and
            id(knowledge.get_fuzzy_vars().base) != id(knowledge_copy.get_fuzzy_vars().base))


def test_to_xml_run():
    # Only test if it doesn't return an exception
    var1 = FuzzyVariable(fuzzy_sets=np.array([TriangularFuzzySet(0, 0.5, 1, 0, "small")], object))
    var2 = FuzzyVariable(fuzzy_sets=np.array([TriangularFuzzySet(0.1, 0.5, 1.1, 1, "small")], object))

    knowledge = Knowledge(np.array([var1, var2], object))
    knowledge.to_xml()

    assert True


def test_plot_fuzzy_variables_no_var():
    knowledge = Knowledge(np.empty(0, object))
    knowledge.plot_fuzzy_variables()


def test_plot_fuzzy_variables_2_vars_1_set():
    var1 = FuzzyVariable(fuzzy_sets=np.array([TriangularFuzzySet(0, 0.5, 1, 0, "small")], object), name="x0")
    var2 = FuzzyVariable(fuzzy_sets=np.array([TriangularFuzzySet(0.1, 0.5, 1.1, 1, "small")], object), domain=np.array([0.0, 1.1]), name="x1")

    knowledge = Knowledge(np.array([var1, var2], object))
    knowledge.plot_fuzzy_variables()


def test_plot_fuzzy_variables_homo_triangle_5():
    knowledge = HomoTriangleKnowledgeFactory_5(1).create()
    knowledge.plot_fuzzy_variables()


def test_plot_fuzzy_variables_homo_triangle_2_3_4_5():
    knowledge = HomoTriangleKnowledgeFactory_2_3_4_5(1).create()
    knowledge.plot_fuzzy_variables()


def test_plot_fuzzy_variables_1_var_4_sets():
    var1 = FuzzyVariable(fuzzy_sets=np.array([DontCareFuzzySet(0),
                                              TriangularFuzzySet(0, 0.5, 1, 1, "medium"),
                                              TriangularFuzzySet(0, 0, 0.5, 0, "small"),
                                              TriangularFuzzySet(0.5, 1, 1, 2, "large")
                                              ], object), name="Petal Size")

    knowledge = Knowledge(np.array([var1], object))
    knowledge.plot_fuzzy_variables()
