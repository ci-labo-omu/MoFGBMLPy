import pytest
import numpy as np
import copy

from mofgbmlpy.fuzzy.knowledge.factory.homo_triangle_knowledge_factory_5 import HomoTriangleKnowledgeFactory_5


def test_none_num_dims():
    with pytest.raises(TypeError):
        HomoTriangleKnowledgeFactory_5(None)


def test_negative_num_dims():
    with pytest.raises(ValueError):
        HomoTriangleKnowledgeFactory_5(-1)


def test_null_num_dims():
    with pytest.raises(ValueError):
        HomoTriangleKnowledgeFactory_5(0)


def test_different_num_dims_var_names_size():
    with pytest.raises(Exception):
        HomoTriangleKnowledgeFactory_5(3, np.array(["x0", "x1"]))

