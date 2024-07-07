import numpy as np
import cython
from mofgbmlpy.fuzzy.fuzzy_term.linguistic_variable import LinguisticVariable
from mofgbmlpy.fuzzy.knowledge.abstract_knowledge_factory cimport AbstractKnowledgeFactory
from mofgbmlpy.fuzzy.knowledge.homo_triangle_knowledge_factory cimport HomoTriangleKnowledgeFactory
from mofgbmlpy.fuzzy.knowledge.knowledge import Knowledge
cimport numpy as cnp
from mofgbmlpy.fuzzy.knowledge.knowledge cimport Knowledge
from mofgbmlpy.fuzzy.fuzzy_term.triangular_fuzzy_set import TriangularFuzzySet


cdef class HomoTriangleKnowledgeFactory_2_3_4_5(HomoTriangleKnowledgeFactory):
    def __init__(self, num_dims, var_names = None):
        num_divisions = np.zeros((num_dims, 4), dtype=np.int_)
        fuzzy_set_names = np.zeros((num_dims, 4), dtype=list)
        for i in range(num_dims):
            num_divisions[i] = np.array([2, 3, 4, 5], dtype=np.int_)
            fuzzy_set_names[i] = np.array([
                ["low_2", "high_2"],
                ["low_3", "medium_3", "high_3"],
                ["low_4", "low_medium_4", "high_medium_4", "high_4"],
                ["very_low_5", "low_5", "medium_5", "high_5", "very_high_5"]],
                dtype=list)

        if var_names is None:
            var_names = np.array([f"x{i}" for i in range(num_dims)], dtype=str)

        super().__init__(num_divisions, var_names, fuzzy_set_names)
