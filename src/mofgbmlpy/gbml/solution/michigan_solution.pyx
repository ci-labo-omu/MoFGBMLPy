import copy

import numpy as np

from mofgbmlpy.data.pattern cimport Pattern
from mofgbmlpy.fuzzy.knowledge.knowledge cimport Knowledge
from mofgbmlpy.fuzzy.rule.rule_builder_core cimport RuleBuilderCore
from mofgbmlpy.fuzzy.rule.abstract_rule cimport AbstractRule
from mofgbmlpy.fuzzy.rule.antecedent.antecedent cimport Antecedent
from mofgbmlpy.fuzzy.rule.consequent.consequent cimport Consequent
from mofgbmlpy.fuzzy.rule.consequent.ruleWeight.abstract_rule_weight cimport AbstractRuleWeight
from mofgbmlpy.gbml.solution.abstract_solution cimport AbstractSolution
cimport numpy as cnp


cdef class MichiganSolution(AbstractSolution):
    def __init__(self, num_objectives, num_constraints, rule_builder, bounds=None, michigan_solution=None, pattern=None):
        if bounds is None:
            bounds = MichiganSolution.make_bounds(knowledge=rule_builder.get_knowledge())

        self._bounds = bounds
        self._rule_builder = rule_builder
        self.__num_wins = 0
        self.__fitness = 0

        super().__init__(len(bounds), num_objectives, num_constraints)

        if michigan_solution is not None:
            self._rule = copy.copy(michigan_solution.get_rule())
            self.set_vars(self._rule.get_antecedent().get_antecedent_indices())
            return

        cnt = 0
        is_rejected = True
        while is_rejected:
            cnt += 1
            self.create_rule(pattern=pattern, michigan_solution=None)
            is_rejected = self._rule.is_rejected_class_label()
            if cnt > 1000:
                # with cython.gil:
                raise Exception("Exceeded maximum number of trials to generate rule")


    @staticmethod
    def make_bounds(knowledge):
        num_dim = knowledge.get_num_dim()

        return np.array([(0, knowledge.get_num_fuzzy_sets(dim_i)-1) for dim_i in range(num_dim)], dtype=object)

    cdef double get_lower_bound(self, int index):
        return self._bounds[index][0]

    cdef double get_upper_bound(self, int index):
        return self._bounds[index][1]

    cdef void create_rule(self, MichiganSolution michigan_solution=None, Pattern pattern=None):
        cdef int[:] antecedent_indices
        if michigan_solution is None:
            antecedent_indices = self._rule_builder.create_antecedent_indices(pattern=pattern, num_rules=1)[0]
            self.set_vars(antecedent_indices)
        else:
            # with cython.gil:
            raise Exception("Not yet implemented")
            # self.set_vars(self._rule_builder.create_antecedent_indices(michigan_solution))  # TODO: Not yet implemented
        self.learning()

    cpdef void learning(self):
        cdef Antecedent antecedent_object
        if self._vars is None:
            # with cython.gil:
            raise Exception("Vars is not defined")

        if self._rule is None or self._rule.get_antecedent() is None:
            antecedent_object = self._rule_builder.create_antecedent_from_indices(self._vars)
            self._rule = self._rule_builder.create(antecedent_object)
        else:
            antecedent_object = self._rule.get_antecedent()
            antecedent_object.set_antecedent_indices(self._vars)
            self._rule.set_consequent(self._rule_builder.create_consequent(antecedent_object))

    cpdef double get_fitness_value(self, cnp.ndarray[double, ndim=1] in_vector):
        return self._rule.get_fitness_value(in_vector)

    cpdef int get_rule_length(self):
        return self._rule.get_rule_length()

    cpdef get_class_label(self):
        return self._rule.get_class_label()

    cdef AbstractRuleWeight get_rule_weight(self):
        return self._rule.get_rule_weight()

    cdef object get_vars_array(self):
        return np.copy(self._vars)

    cpdef AbstractRule get_rule(self):
        return self._rule

    cdef RuleBuilderCore get_rule_builder(self):
        return self._rule_builder

    cpdef Consequent get_consequent(self):
        return self._rule.get_consequent()

    cpdef Antecedent get_antecedent(self):
        return self._rule.get_antecedent()

    cdef cnp.ndarray[double, ndim=1] get_compatible_grade(self, cnp.ndarray[double, ndim=1] attribute_vector):
        return self._rule.get_compatible_grade(attribute_vector)

    cdef double get_compatible_grade_value(self, cnp.ndarray[double, ndim=1] attribute_vector):
        return self._rule.get_compatible_grade_value(attribute_vector)

    def __copy__(self):
        return MichiganSolution(self.get_num_objectives(), self.get_num_constraints(), copy.copy(self._rule_builder), michigan_solution=self)

    cpdef double compute_coverage(self):
        coverage = 1
        dim_i = 0
        for fuzzy_set_id in self._vars:
            coverage *= self._rule_builder.get_knowledge().get_support(dim_i, fuzzy_set_id)
            dim_i += 1
        return coverage

    cpdef void reset_num_wins(self):
        self.__num_wins = 0

    cpdef void reset_fitness(self):
        self.__fitness = 0

    cpdef void inc_num_wins(self):
        self.__num_wins += 1

    cpdef void inc_fitness(self):
        self.__fitness += 1

    cpdef int get_num_wins(self):
        return self.__num_wins

    cpdef int get_fitness(self):
        return self.__fitness

    def __repr__(self):
        return f"(MichiganSolution) {self._rule}"
