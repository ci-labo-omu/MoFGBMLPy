import copy
import time

import numpy as np

from fuzzy.knowledge.knowledge import Knowledge
from gbml.solution.abstract_solution import AbstractSolution
cimport numpy as cnp


class MichiganSolution(AbstractSolution):
    _bounds = None
    _rule = None
    _rule_builder = None

    def __init__(self, num_objectives, num_constraints, rule_builder, bounds=None, michigan_solution=None, pattern=None):
        if bounds is None:
            bounds = MichiganSolution.make_bounds(knowledge=rule_builder.get_knowledge())

        self._bounds = bounds
        self._rule_builder = rule_builder

        super().__init__(len(bounds), num_objectives, num_constraints)

        if michigan_solution is not None:
            self._rule = copy.copy(michigan_solution.get_rule())
            self.set_vars(self._rule.get_antecedent().get_antecedent_indices())
            return

        cnt = 0
        is_rejected = True
        while is_rejected:
            cnt += 1
            self.create_rule(pattern=pattern)
            is_rejected = self._rule.is_rejected_class_label()
            if cnt > 1000:
                raise Exception("Exceeded maximum number of trials to generate rule")

    @staticmethod
    def make_bounds(knowledge):
        num_dim = knowledge.get_num_dim()

        return np.array([(0, knowledge.get_num_fuzzy_sets(dim_i)-1) for dim_i in range(num_dim)], dtype=object)

    def get_lower_bound(self, index):
        return self._bounds[index][0]

    def get_upper_bound(self, index):
        return self._bounds[index][1]

    def create_rule(self, michigan_solution=None, pattern=None):
        if michigan_solution is None:
            antecedent_indices = self._rule_builder.create_antecedent_indices(pattern=pattern)
            self.set_vars(antecedent_indices)
        else:
            raise Exception("Not yet implemented")
            # self.set_vars(self._rule_builder.create_antecedent_indices(michigan_solution))  # TODO: Not yet implemented
        self.learning()

    def learning(self):
        if self._vars is None:
            raise Exception("Vars is not defined")

        if self._rule is None or self._rule.get_antecedent() is None:
            antecedent_object = self._rule_builder.create_antecedent_from_indices(self._vars)
            self._rule = self._rule_builder.create(antecedent_object)
        else:
            antecedent_object = self._rule.get_antecedent()
            antecedent_object.set_antecedent_indices(self._vars)
            self._rule.set_consequent(self._rule_builder.create_consequent(antecedent_object))

    def get_fitness_value(self, in_vector):
        return self._rule.get_fitness_value(in_vector)

    def get_rule_length(self):
        return self._rule.get_rule_length()

    def get_class_label(self):
        return self._rule.get_class_label()

    def get_rule_weight(self):
        return self._rule.get_rule_weight()

    def get_vars_array(self):
        return np.copy(self._vars)

    def get_rule(self):
        return self._rule

    def get_rule_builder(self):
        return self._rule_builder

    def get_consequent(self):
        return self._rule.get_consequent()

    def get_antecedent(self):
        return self._rule.get_antecedent()

    def get_compatible_grade(self, attribute_vector):
        return self._rule.get_compatible_grade(attribute_vector)

    def get_compatible_grade_value(self, attribute_vector):
        return self._rule.get_compatible_grade_value(attribute_vector)

    def __copy__(self):
        return MichiganSolution(self.get_num_objectives(), self.get_num_constraints(), copy.copy(self._rule_builder), michigan_solution=self)

    def compute_coverage(self):
        coverage = 1
        dim_i = 0
        for fuzzy_set_id in self._vars:
            coverage *= self._rule_builder.get_knowledge().get_support(dim_i, fuzzy_set_id)
            dim_i += 1
        return coverage

    class MichiganSolutionBuilder(AbstractSolution.SolutionBuilderCore):
        def __init__(self, bounds, num_objectives, num_constraints, rule_builder):
            super().__init__(bounds, num_objectives, num_constraints, rule_builder)

        def create(self, num_solutions=1, pattern=None):
            cdef cnp.ndarray[object, ndim=1] solutions = np.empty(num_solutions, dtype=object)
            cdef int i
            bounds = self._bounds

            if bounds is None:
                bounds = MichiganSolution.make_bounds(self._rule_builder.get_knowledge())


            for i in range(num_solutions):
                solutions[i] = MichiganSolution(self._num_objectives, self._num_constraints, self._rule_builder, bounds, pattern=pattern)
                # solutions[i].set_attribute(attribute_id, 0) # TODO: check usage in java version
                # solutions[i].set_attribute(attribute_id_fitness, 0)


            return solutions

        def __copy__(self):
            return MichiganSolution.MichiganSolutionBuilder(self._bounds, self._num_objectives, self._num_constraints, copy.copy(self._rule_builder))