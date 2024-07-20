import numpy as np

from mofgbmlpy.fuzzy.fuzzy_term.fuzzy_set.dont_care_fuzzy_set import DontCareFuzzySet
from mofgbmlpy.fuzzy.fuzzy_term.fuzzy_variable import FuzzyVariable
from mofgbmlpy.fuzzy.knowledge.factory.abstract_knowledge_factory cimport AbstractKnowledgeFactory
from mofgbmlpy.fuzzy.knowledge.knowledge import Knowledge
cimport numpy as cnp
from mofgbmlpy.fuzzy.knowledge.knowledge cimport Knowledge
from mofgbmlpy.fuzzy.fuzzy_term.fuzzy_set.triangular_fuzzy_set import TriangularFuzzySet

cdef class HomoTriangleKnowledgeFactory(AbstractKnowledgeFactory):
    def __init__(self, int[:,:] num_divisions, var_names, fuzzy_set_names):
        if (num_divisions is None or len(num_divisions) == 0 or
                var_names is None or len(var_names) == 0 or
                fuzzy_set_names is None or len(fuzzy_set_names) == 0):
            raise Exception("Parameters can't be null or empty")

        if len(num_divisions.base.shape) != 2 or num_divisions.shape[1] == 0:
            raise Exception("num_divisions second dimension can't be null")

        num_dims = num_divisions.shape[0]

        if len(var_names) != num_dims or len(fuzzy_set_names) != num_dims:
            raise Exception("var_names and num_divisions first dimension must be of the same size as num_division first one")

        for item in var_names:
            if item is None:
                raise Exception("Var names can't be None")

        for dim_i in range(len(num_divisions)):
            num_divisions_dim_i = len(num_divisions[dim_i])
            if len(fuzzy_set_names[dim_i]) != num_divisions_dim_i:
                raise Exception(f"fuzzy_set_names second dimension is invalid, got {fuzzy_set_names.shape[1]} but expected {num_divisions_dim_i}")
            for j in range(num_divisions_dim_i):
                if len(fuzzy_set_names[dim_i][j]) != num_divisions[dim_i][j]:
                    raise Exception(f"fuzzy_set_names third dimension is invalid, got {len(fuzzy_set_names[dim_i][j])} but expected {num_divisions[dim_i][j]}")
                if num_divisions[dim_i][j] <= 0:
                    raise Exception("num_divisions can't contain null or negative values")

        self.__num_divisions = num_divisions
        self.__var_names = var_names
        self.__fuzzy_set_names = fuzzy_set_names
    
    @staticmethod
    def make_triangle_knowledge_params(int num_partitions):
        cdef int i
        cdef double left
        cdef double center
        cdef double right
        cdef cnp.ndarray[double, ndim=2] params
        cdef double[:] partition

        if num_partitions <= 1:
            raise Exception("num_partitions can't be lesser or equal to 1")

        params = np.zeros((num_partitions, 3))
        partition = np.zeros(num_partitions+1)

        # e.g.: K = 2: 0, 1/2, 1
        # e.g.: K = 3: 0, 1/4, 3/4, 1
        # e.g.: K = 5: 0, 1/8, 3/8, 5/8, 7/8, 1

        for i in range(1, num_partitions):
            partition[i] = (2*i-1) / ((num_partitions-1) * 2)

        partition[num_partitions] = 1

        for i in range(num_partitions):
            if i == 0:  # 1st partition
                params[i] = np.array([0, 0, 2*partition[1]])
            elif i == partition.shape[0]-2:  # last partition
                params[i] = np.array([2*partition[i]-1, 1, 1])
            elif i>0 and i<partition.shape[0]-2:  # If the index is valid
                left = partition[i]*3/2 - partition[i+1]/2
                center = (partition[i] + partition[i + 1]) / 2
                right = partition[i+1]*3/2 - partition[i]/2
                params[i] = np.array([left, center, right])

        return params

    cpdef create(self):
        cdef Knowledge knowledge

        cdef int set_id = 0
        cdef int dim_i
        cdef int j

        knowledge = Knowledge()
        fuzzy_sets = np.empty(len(self.__num_divisions), dtype=object)

        for dim_i in range(len(self.__num_divisions)):
            current_support_values = [1]
            current_set = [DontCareFuzzySet(id=set_id)]
            set_id += 1

            for j in range(len(self.__num_divisions[dim_i])):
                params = HomoTriangleKnowledgeFactory.make_triangle_knowledge_params(self.__num_divisions[dim_i][j])
                for div_i in range(self.__num_divisions[dim_i][j]):
                    new_fuzzy_set = TriangularFuzzySet(left=params[div_i][0], center=params[div_i][1], right=params[div_i][2], id=set_id, term=self.__fuzzy_set_names[dim_i][j][div_i])
                    set_id += 1
                    current_set.append(new_fuzzy_set)

                    if div_i == 0 or self.__num_divisions[dim_i][j] <= 1: # DC or one division
                        current_support_values.append(1)
                    elif div_i == 1 or div_i == self.__num_divisions[dim_i][j] - 1:
                        current_support_values.append(1 / (self.__num_divisions[dim_i][j] - 1))
                    else:
                        current_support_values.append(2 / (self.__num_divisions[dim_i][j] - 1))

            fuzzy_sets[dim_i] = FuzzyVariable(np.array(current_set, dtype=object), np.array(current_support_values), str(self.__var_names[dim_i]))
        knowledge.set_fuzzy_vars(fuzzy_sets)
        return knowledge