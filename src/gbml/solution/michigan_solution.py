import numpy as np

from src.gbml.solution.abstract_solution import AbstractSolution
from src.fuzzy.knowledge.knowledge import Knowledge


class MichiganSolution(AbstractSolution):
    _bounds = None
    _rule = None
    _rule_builder = None

    def __init__(self, num_objectives, num_constraints, rule_builder, bounds=None, michigan_solution=None):
        if bounds is None:
            bounds = MichiganSolution.make_bounds()

        super().__init__(len(bounds), num_objectives, num_constraints)
        self._bounds = bounds
        self._rule_builder = rule_builder

        cnt = 0
        is_rejected = True
        while is_rejected:
            cnt += 1
            self.create_rule()
            if cnt > 1000:
                raise Exception("Exceeded maximum number of trials to generate rule")

        if michigan_solution is not None:
            self.create_rule(michigan_solution)
            while self._rule.is_rejected():
                self.create_rule()

    def make_bounds(self):
        knowledge = Knowledge.get_instance()
        num_dim = knowledge.get_num_dim()

        return np.array([(0, len(knowledge.get_fuzzy_set(dim_i))-1) for dim_i in range(num_dim)], dtype=object)

    def get_lower_bound(self, index):
        return self._bounds[index][0]

    def get_upper_bound(self, index):
        return self._bounds[index][1]

    def set_vars(self, new_vars):
        for i in range(len(new_vars)):
            self.set_var(i, new_vars[i])

    def get_var(self, index):
        return self._vars[index]

    def create_rule(self, michigan_solution=None):
        if michigan_solution is None:
            antecedent_indices = self._rule_builder.create_antecedent_indices()
            self.set_vars(antecedent_indices)
        else:
            self.set_vars(self._rule_builder.create_antecedent_indices(michigan_solution))
        self.learning()

    def learning(self):
        if self._vars is None:
            raise Exception("Vars is not defined")
        self._rule = self._rule_builder.create_consequent(self.get_vars_array())

    def get_fitness_value(self, in_vector):
        self._rule.get_fitness_value(self.get_vars_array(), in_vector)

    def get_rule_length(self):
        return self._rule.get_rule_length(self.get_vars_array())

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

    # TODO: define builder ?