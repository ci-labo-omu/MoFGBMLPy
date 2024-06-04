import numpy as np

from fuzzy.knowledge.knowledge import Knowledge
from src.fuzzy.fuzzy_term.fuzzy_term_triangular import FuzzyTermTriangular
from src.fuzzy.fuzzy_term.fuzzy_term_dont_care import FuzzyTermDontCare


class HomoTriangleKnowledgeFactory:
    @staticmethod
    def make_triangle_knowledge_params(num_partitions):
        params = np.zeros((num_partitions, 3))
        partition = np.zeros(num_partitions+1)  # e.g.: K = 5: 0, 1/8, 3/8, 5/8, 7/8, 1

        # TODO: as in the Java version we should save this values so that we don't recompute them if needed
        for i in range(1, num_partitions):
            partition[i] = (2*i-1) / ((num_partitions-1) * 2)

        partition[num_partitions] = 1

        for i in range(num_partitions):
            if i == 0:  # 1st partition
                params[i] = np.array([0, 0, 2*partition[i+1]])
            elif i == len(partition)-2:  # last partition
                params[i] = np.array([2*partition[i]-1, 1, 1])
            elif i>0 and i<len(partition)-2:  # If the index is valid
                left = partition[i]*3/2 - partition[i+1]/2
                center = (partition[i] + partition[i + 1]) / 2
                right = partition[i+1]*3/2 - partition[i]/2
                params[i] = np.array([left, center, right])

        return params

    @staticmethod
    def create(num_divisions):
        fuzzy_sets = []

        for dim_i in range(len(num_divisions)):
            current_set = [FuzzyTermDontCare()]

            for j in range(len(num_divisions[dim_i])):
                params = HomoTriangleKnowledgeFactory.make_triangle_knowledge_params(num_divisions[dim_i][j])
                for div_i in range(num_divisions[dim_i][j]):
                    current_set.append(FuzzyTermTriangular(params[div_i][0], params[div_i][1], params[div_i][2]))

            fuzzy_sets.append(current_set)

        Knowledge.get_instance().set_fuzzy_sets(fuzzy_sets)

    @staticmethod
    def create2_3_4_5(num_dims):
        num_divisions = np.zeros((num_dims, 4), dtype=np.int_)
        for i in range(num_dims):
            num_divisions[i] = np.array([2, 3, 4, 5], dtype=np.int_)

        HomoTriangleKnowledgeFactory.create(num_divisions)
