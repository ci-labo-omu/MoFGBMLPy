import numpy as np
from mofgbmlpy.fuzzy.knowledge.factory.homo_triangle_knowledge_factory cimport HomoTriangleKnowledgeFactory

cdef class HomoTriangleKnowledgeFactory_5(HomoTriangleKnowledgeFactory):
    """Helper class to create a Knowledge object with triangular fuzzy sets with 6 fuzzy sets per variable (the same ones): DC + 5"""
    def __init__(self, int num_dims, var_names = None):
        """Constructor

        Args:
            num_dims (int): Number of dimensions (i.e. number of variables)
            var_names (str[]): Array of the variables name. e.g.: ["x0", "x1"]
        """
        if num_dims is None or num_dims <= 0:
            raise Exception("num_dims must be a positive integer")

        num_divisions = np.zeros((num_dims, 1), dtype=np.int_)
        fuzzy_set_names = np.zeros((num_dims, 1, 5), dtype=object)

        for i in range(num_dims):
            num_divisions[i] = np.array([5], dtype=np.int_)
            fuzzy_set_names[i] = np.array([
                ["very_low_5", "low_5", "medium_5", "high_5", "very_high_5"]],
                dtype=list)

        if var_names is None:
            var_names = np.array([f"x{i}" for i in range(num_dims)], dtype=str)
        elif len(var_names) != num_dims:
            raise Exception(f"var_names must be of a size equals to num_dims (i.e. {num_dims})")

        super().__init__(num_divisions, var_names, fuzzy_set_names)
