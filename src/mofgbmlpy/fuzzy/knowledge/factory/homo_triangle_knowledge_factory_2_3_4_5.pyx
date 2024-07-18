import numpy as np
from mofgbmlpy.fuzzy.knowledge.factory.homo_triangle_knowledge_factory cimport HomoTriangleKnowledgeFactory

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
