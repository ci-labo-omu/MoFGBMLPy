import numpy as np

from src.fuzzy.fuzzy_term.dont_care_fuzzy_set import DontCareFuzzySet
from src.fuzzy.fuzzy_term.linguistic_variable_mofgbml import LinguisticVariableMoFGBML
from src.fuzzy.knowledge.knowledge import Knowledge
from simpful import TriangleFuzzySet


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
    def create(num_divisions, var_names, fuzzy_set_names):
        knowledge = Knowledge()
        fuzzy_sets = []

        for dim_i in range(len(num_divisions)):
            current_set = [DontCareFuzzySet()]

            for j in range(len(num_divisions[dim_i])):
                params = HomoTriangleKnowledgeFactory.make_triangle_knowledge_params(num_divisions[dim_i][j])
                for div_i in range(num_divisions[dim_i][j]):
                    new_fuzzy_set = TriangleFuzzySet(a=params[div_i][0], b=params[div_i][1], c=params[div_i][2], term=fuzzy_set_names[dim_i][j][div_i])
                    current_set.append(new_fuzzy_set)

            fuzzy_sets.append(LinguisticVariableMoFGBML(current_set, var_names[dim_i]))

        knowledge.set_fuzzy_sets(fuzzy_sets)
        return knowledge

    @staticmethod
    def create2_3_4_5(num_dims, var_names=None):
        num_divisions = np.zeros((num_dims, 4), dtype=np.int_)
        fuzzy_set_names = np.zeros((num_dims, 4), dtype=list)
        for i in range(num_dims):
            num_divisions[i] = np.array([2, 3, 4, 5], dtype=np.int_)
            fuzzy_set_names[i] = np.array([
                ["low", "high"],
                ["low", "medium", "high"],
                ["low", "low_medium", "high_medium", "high"],
                ["very_low", "low", "medium", "high", "very_high"]],
                dtype=list)

        if var_names is None:
            var_names = np.array([f"x{i}" for i in range(num_dims)], dtype=str)

        return HomoTriangleKnowledgeFactory.create(num_divisions, var_names, fuzzy_set_names)
