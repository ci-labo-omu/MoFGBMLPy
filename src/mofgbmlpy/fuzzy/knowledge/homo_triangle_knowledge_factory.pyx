import numpy as np
import cython
from mofgbmlpy.fuzzy.fuzzy_term.dont_care_fuzzy_set import DontCareFuzzySet
from mofgbmlpy.fuzzy.fuzzy_term.linguistic_variable import LinguisticVariable
from mofgbmlpy.fuzzy.knowledge.knowledge import Knowledge
cimport numpy as cnp
from mofgbmlpy.fuzzy.knowledge.knowledge cimport Knowledge
from mofgbmlpy.fuzzy.fuzzy_term.triangular_fuzzy_set import TriangularFuzzySet

cdef class HomoTriangleKnowledgeFactory:
    @staticmethod
    def make_triangle_knowledge_params(num_partitions):
        cdef int i
        cdef double left
        cdef double center
        cdef double right
        cdef cnp.ndarray[double, ndim=2] params
        cdef cnp.ndarray[double, ndim=1] partition

        params = np.zeros((num_partitions, 3))
        partition = np.zeros(num_partitions+1)  # e.g.: K = 5: 0, 1/8, 3/8, 5/8, 7/8, 1

        for i in range(1, num_partitions):
            partition[i] = (2*i-1) / ((num_partitions-1) * 2)

        partition[num_partitions] = 1

        for i in range(num_partitions):
            if i == 0:  # 1st partition
                params[i] = np.array([0, 0, 2*partition[i+1]])
            elif i == partition.size-2:  # last partition
                params[i] = np.array([2*partition[i]-1, 1, 1])
            elif i>0 and i<partition.size-2:  # If the index is valid
                left = partition[i]*3/2 - partition[i+1]/2
                center = (partition[i] + partition[i + 1]) / 2
                right = partition[i+1]*3/2 - partition[i]/2
                params[i] = np.array([left, center, right])

        return params

    @staticmethod
    def create(num_divisions, var_names, fuzzy_set_names):
        cdef Knowledge knowledge

        knowledge = Knowledge()
        fuzzy_sets = np.empty(len(num_divisions), dtype=object)
        #TODO: use numpy
        for dim_i in range(len(num_divisions)):
            current_support_values = [1]
            current_set = [DontCareFuzzySet()]
            for j in range(len(num_divisions[dim_i])):
                params = HomoTriangleKnowledgeFactory.make_triangle_knowledge_params(num_divisions[dim_i][j])
                for div_i in range(num_divisions[dim_i][j]):
                    new_fuzzy_set = TriangularFuzzySet(left=params[div_i][0], center=params[div_i][1], right=params[div_i][2], term=fuzzy_set_names[dim_i][j][div_i])
                    current_set.append(new_fuzzy_set)

                    if div_i == 0 or num_divisions[dim_i][j] <= 1: # DC or one division
                        current_support_values.append(1)
                    elif div_i == 1 or div_i == num_divisions[dim_i][j] - 1:
                        current_support_values.append(1 / (num_divisions[dim_i][j] - 1))
                    else:
                        current_support_values.append(2 / (num_divisions[dim_i][j] - 1))

            fuzzy_sets[dim_i] = LinguisticVariable(current_set, var_names[dim_i], current_support_values)

        knowledge.set_fuzzy_sets(fuzzy_sets)
        return knowledge

    @staticmethod
    def create2_3_4_5(num_dims, var_names=None):
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

        return HomoTriangleKnowledgeFactory.create(num_divisions, var_names, fuzzy_set_names)
