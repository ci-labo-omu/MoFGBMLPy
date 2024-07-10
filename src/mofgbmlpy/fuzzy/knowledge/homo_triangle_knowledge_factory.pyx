import numpy as np
import cython

from mofgbmlpy.fuzzy.fuzzy_term.dont_care_fuzzy_set import DontCareFuzzySet
from mofgbmlpy.fuzzy.fuzzy_term.linguistic_variable import LinguisticVariable
from mofgbmlpy.fuzzy.knowledge.abstract_knowledge_factory cimport AbstractKnowledgeFactory
from mofgbmlpy.fuzzy.knowledge.knowledge import Knowledge
cimport numpy as cnp
from mofgbmlpy.fuzzy.knowledge.knowledge cimport Knowledge
from mofgbmlpy.fuzzy.fuzzy_term.triangular_fuzzy_set import TriangularFuzzySet

cdef class HomoTriangleKnowledgeFactory(AbstractKnowledgeFactory):
    def __init__(self, num_divisions, var_names, fuzzy_set_names):
        self.__num_divisions = num_divisions
        self.__var_names = var_names
        self.__fuzzy_set_names = fuzzy_set_names
    
    @staticmethod
    def make_triangle_knowledge_params(num_partitions):
        cdef int i
        cdef double left
        cdef double center
        cdef double right
        cdef cnp.ndarray[double, ndim=2] params
        cdef double[:] partition
        cdef int num_partitions_int = num_partitions

        params = np.zeros((num_partitions_int, 3))
        partition = np.zeros(num_partitions_int+1)  # e.g.: K = 5: 0, 1/8, 3/8, 5/8, 7/8, 1

        for i in range(1, num_partitions_int):
            partition[i] = (2*i-1) / ((num_partitions_int-1) * 2)

        partition[num_partitions_int] = 1

        for i in range(num_partitions_int):
            if i == 0:  # 1st partition
                params[i] = np.array([0, 0, 2*partition[i+1]])
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

        knowledge = Knowledge()
        fuzzy_sets = np.empty(len(self.__num_divisions), dtype=object)

        #TODO: use numpy
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

            fuzzy_sets[dim_i] = LinguisticVariable(np.array(current_set, dtype=object), self.__var_names[dim_i], np.array(current_support_values))
        knowledge.set_fuzzy_sets(fuzzy_sets)
        return knowledge