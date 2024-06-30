import numpy as np
from libc.stdlib cimport qsort
from pymoo.core.population import Population

from mofgbmlpy.gbml.solution.michigan_solution cimport MichiganSolution

cdef class RuleStyleSurvival:
    @staticmethod
    def sort_by_fitness(arr):
        arr = sorted(arr, key=lambda x: x.get_fitness(), reverse=True)
        return np.array(arr)

    @staticmethod
    def replace(pop, offspring, max_num_rules):
        # Check if we already exceed max num rules
        num_replacements = 0
        if max_num_rules < len(pop) + len(offspring):
            num_replacements = len(pop) + len(offspring) - max_num_rules

        new_shape = (pop.shape[0] + len(offspring) - num_replacements,) + pop.shape[1:]
        new_pop = np.empty(new_shape, dtype=object)

        # TODO: remove fitness class attribute in michigan solution and use objectives array instead

        # Sort by fitness (objective 0)
        RuleStyleSurvival.sort_by_fitness(pop)

        # Copy individuals that won't be replaced
        for i in range(len(pop)-num_replacements):
            new_pop[i] = pop[i]

        # Replace the worst individuals in the current population with the offspring and add rules
        k = 0
        for i in range(len(pop)-num_replacements, len(new_pop)):
            new_pop[i] = offspring[k]
            k += 1

        return new_pop
