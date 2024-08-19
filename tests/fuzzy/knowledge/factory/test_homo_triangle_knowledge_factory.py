import pytest
import numpy as np
import copy

from mofgbmlpy.fuzzy.knowledge.factory.homo_triangle_knowledge_factory import HomoTriangleKnowledgeFactory


def test_none_num_divisions():
    num_divisions = None
    var_names = np.array(["var1", "var2"])
    fuzzy_set_names = np.array([
        np.array([["normal_1"], ["low_1", "high_1"]], list),
        np.array([["normal_2"], ["low_2", "medium_2", "high_2"]], list),
    ])
    with pytest.raises(TypeError):
        HomoTriangleKnowledgeFactory(num_divisions, var_names, fuzzy_set_names)


def test_empty_num_divisions():
    num_divisions = np.empty(0, int)
    var_names = np.array(["var1", "var2"])
    fuzzy_set_names = np.array([
        np.array([["normal_1"], ["low_1", "high_1"]], list),
        np.array([["normal_2"], ["low_2", "medium_2", "high_2"]], list),
    ])
    with pytest.raises(Exception):
        HomoTriangleKnowledgeFactory(num_divisions, var_names, fuzzy_set_names)


def test_none_var_names():
    num_divisions = np.array([[1, 2], [1, 3]])
    var_names = None
    fuzzy_set_names = np.array([
        np.array([["normal_1"], ["low_1", "high_1"]], list),
        np.array([["normal_2"], ["low_2", "medium_2", "high_2"]], list),
    ])
    with pytest.raises(TypeError):
        HomoTriangleKnowledgeFactory(num_divisions, var_names, fuzzy_set_names)


def test_empty_var_names():
    num_divisions = np.array([[1, 2], [1, 3]])
    var_names = np.empty(0, str)
    fuzzy_set_names = np.array([
        np.array([["normal_1"], ["low_1", "high_1"]], list),
        np.array([["normal_2"], ["low_2", "medium_2", "high_2"]], list),
    ])
    with pytest.raises(ValueError):

        HomoTriangleKnowledgeFactory(num_divisions, var_names, fuzzy_set_names)


def test_none_fuzzy_set_names():
    num_divisions = np.array([[1, 2], [1, 3]])
    var_names = np.array(["var1", "var2"])
    fuzzy_set_names = None
    with pytest.raises(TypeError):

        HomoTriangleKnowledgeFactory(num_divisions, var_names, fuzzy_set_names)


def test_empty_fuzzy_set_names():
    num_divisions = np.array([[1, 2], [1, 3]])
    var_names = np.array(["var1", "var2"])
    fuzzy_set_names = np.empty(0, str)
    with pytest.raises(ValueError):

        HomoTriangleKnowledgeFactory(num_divisions, var_names, fuzzy_set_names)


def test_negative_num_divisions():
    num_divisions = np.array([[1, -2], [1, 3]])
    var_names = np.array(["var1", "var2"])
    fuzzy_set_names = np.array([
        np.array([["normal_1"], ["low_1", "high_1"]], list),
        np.array([["normal_2"], ["low_2", "medium_2", "high_2"]], list),
    ])

    with pytest.raises(ValueError):
        HomoTriangleKnowledgeFactory(num_divisions, var_names, fuzzy_set_names)


def test_null_num_divisions():
    num_divisions = np.array([[1, 2], [0, 3]])
    var_names = np.array(["var1", "var2"])
    fuzzy_set_names = np.array([
        np.array([["normal_1"], ["low_1", "high_1"]], list),
        np.array([["normal_2"], ["low_2", "medium_2", "high_2"]], list),
    ])
    with pytest.raises(ValueError):
        HomoTriangleKnowledgeFactory(num_divisions, var_names, fuzzy_set_names)


def test_invalid_shape_var_names():
    num_divisions = np.array([[1, 2], [1, 3]])
    var_names = np.array(["var1", "var2", "var2"])
    fuzzy_set_names = np.array([
        np.array([["normal_1"], ["low_1", "high_1"]], list),
        np.array([["normal_2"], ["low_2", "medium_2", "high_2"]], list),
    ])
    with pytest.raises(ValueError):
        HomoTriangleKnowledgeFactory(num_divisions, var_names, fuzzy_set_names)


def test_invalid_var_names_contains_none():
    num_divisions = np.array([[1, 2], [1, 3]])
    var_names = np.array(["var1", None])
    fuzzy_set_names = np.array([
        np.array([["normal_1"], ["low_1", "high_1"]], list),
        np.array([["normal_2"], ["low_2", "medium_2", "high_2"]], list),
    ])
    with pytest.raises(TypeError):
        HomoTriangleKnowledgeFactory(num_divisions, var_names, fuzzy_set_names)


def test_invalid_shape_fuzzy_set_names_dim1():
    num_divisions = np.array([[1, 2], [1, 3]])
    var_names = np.array(["var1", "var2"])
    fuzzy_set_names = np.array([
        np.array([["normal_1"], ["low_1", "high_1"]], list),
        np.array([["normal_2"], ["low_2", "medium_2", "high_2"]], list),
        np.array([["normal_2"], ["low_2", "medium_2", "high_2"]], list),
    ])
    with pytest.raises(ValueError):
        HomoTriangleKnowledgeFactory(num_divisions, var_names, fuzzy_set_names)


def test_invalid_shape_fuzzy_set_names_dim2():
    num_divisions = np.array([[1, 3], [1, 3]])
    var_names = np.array(["var1", "var2"])
    fuzzy_set_names = np.array([
        np.array([["normal_1"], ["low_1", "high_1"]], list),
        np.array([["normal_2"], ["low_2", "medium_2", "high_2"]], list),
    ])
    with pytest.raises(ValueError):
        HomoTriangleKnowledgeFactory(num_divisions, var_names, fuzzy_set_names)


def test_invalid_shape_fuzzy_set_names_dim3():
    num_divisions = np.array([[1, 2], [1, 3]])
    var_names = np.array(["var1", "var2"])
    fuzzy_set_names = np.array([
        np.array([["normal_1"], ["low_1", "high_1", "very_high_1"]], list),
        np.array([["normal_2"], ["low_2", "medium_2", "high_2"]], list),
    ])
    with pytest.raises(ValueError):
        HomoTriangleKnowledgeFactory(num_divisions, var_names, fuzzy_set_names)


def test_make_triangle_knowledge_params_negative_num_partitions():
    with pytest.raises(ValueError):
        HomoTriangleKnowledgeFactory.make_triangle_knowledge_params(-1)


def test_make_triangle_knowledge_params_null_num_partitions():
    with pytest.raises(ValueError):
        HomoTriangleKnowledgeFactory.make_triangle_knowledge_params(0)


@pytest.mark.parametrize("num_partition", np.random.randint(2, 10, 5))
def test_make_triangle_knowledge_params_valid(num_partition):
    params = HomoTriangleKnowledgeFactory.make_triangle_knowledge_params(num_partition)
    precision = 1e-6

    assert params.shape[0] == num_partition and params.shape[1] == 3

    half_triangle_base_size = 1/(num_partition-1)

    # TODO: Maybe use this method in the HomoTriangleKnowledgeFactory instead of the current one as it's more readable
    for i in range(num_partition):
        if i == 0:
            start = 0
            middle = start
            end = half_triangle_base_size
        elif i == num_partition-1:
            start = (i-1) * half_triangle_base_size
            middle = 1
            end = middle
        else:
            start = (i - 1) * half_triangle_base_size
            middle = i * half_triangle_base_size
            end = (i + 1) * half_triangle_base_size
        assert (abs(params[i][0] - start) < precision and
                abs(params[i][1] - middle) < precision and
                abs(params[i][2] - end) < precision)


def create_factory(num_divisions):
    fuzzy_set_names = np.empty(len(num_divisions), dtype=list)

    for i in range(len(num_divisions)):
        fuzzy_set_names[i] = []
        for j in range(len(num_divisions[i])):
            items = []

            for k in range(num_divisions[i][j]):
                items.append(f"v_{i}_{j}_{k}")
            fuzzy_set_names[i].append(items)

    var_names = np.array([f"x{i}" for i in range(len(num_divisions))], dtype=str)

    return HomoTriangleKnowledgeFactory(num_divisions, var_names, fuzzy_set_names)


@pytest.mark.parametrize("exec_run", [i for i in range(10)])
def test_create(exec_run):
    num_divisions = np.random.randint(2, 7, (5, 2))
    factory = create_factory(num_divisions)
    knowledge = factory.create()
    assert knowledge.get_num_dim() == len(num_divisions)

    for i in range(len(num_divisions)):
        num_fuzzy_sets = 1  # Don't care fuzzy set
        for num_partitions in num_divisions[i]:
            num_fuzzy_sets += num_partitions

        assert knowledge.get_num_fuzzy_sets(i) == num_fuzzy_sets
