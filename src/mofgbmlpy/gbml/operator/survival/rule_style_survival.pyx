import numpy as np
from libc.stdlib cimport qsort
from pymoo.core.population import Population

from mofgbmlpy.gbml.solution.michigan_solution cimport MichiganSolution

cdef class RuleStyleSurvival:
    """Static methods used to replace a population of Michigan rules with its offsprings (survival step of the genetic algorithm)"""
    @staticmethod
    def sort_by_fitness(arr):
        """Sort the array of Michigan solutions by the their fitness

        Args:
            arr (MichiganSolution[]): Array of michigan solutions

        Returns:
            MichiganSolution[]: Sorted array
        """
        arr = sorted(arr, key=lambda x: x.get_fitness(), reverse=True)
        return np.array(arr)

    @staticmethod
    def replace(pop, offspring, max_num_rules):
        """Replace a population of Michigan solutions by an offspring population in such a way that all offsprings are added and to avoid going above the max number of rules we remove the worst solutions in the initial population (based on fitness)

        Args:
            pop ():
            offspring ():
            max_num_rules ():

        Returns:

        """
        # Check if we already exceed max num rules
        num_replacements = 0
        if max_num_rules < len(pop) + len(offspring):
            num_replacements = len(pop) + len(offspring) - max_num_rules

        new_shape = (pop.shape[0] + len(offspring) - num_replacements,) + pop.shape[1:]
        new_pop = np.empty(new_shape, dtype=object)

        # Sort by fitness if we need to replace the worst individuals (if not we just have to append to the current pop)
        if num_replacements > 0:
            pop = RuleStyleSurvival.sort_by_fitness(pop)

        # Copy individuals that won't be replaced
        for i in range(len(pop)-num_replacements):
            new_pop[i] = pop[i]

        # Replace the worst individuals in the current population with the offspring and add rules if there is still space
        k = 0
        for i in range(len(pop)-num_replacements, len(new_pop)):
            new_pop[i] = offspring[k]
            k += 1

        return new_pop
