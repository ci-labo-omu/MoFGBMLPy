import numpy as np
from src.fuzzy.knowledge.knowledge import Context


class Antecedent:
    __antecedent_indices = None

    def __init__(self, antecedent_indices):
        self.__antecedent_indices = antecedent_indices

    def get_antecedent_length(self):
        return len(self.__antecedent_indices)

    def get_antecedent_indices(self):
        return self.__antecedent_indices

    def get_compatible_grade(self, attribute_vector):
        # compute membership value
        grade = np.zeros(self.get_antecedent_length())

        if self.get_antecedent_length() != attribute_vector.get_num_dim():
            raise Exception("antecedent_indices and attribute_vector must have the same length")

        for i in range(attribute_vector.get_num_dim()):
            val = attribute_vector.get_value(i)
            if self.__antecedent_indices[i] < 0 and val < 0:
                # categorical
                grade[i] = 1.0 if self.__antecedent_indices[i] == round(val) else 0.0
            elif self.__antecedent_indices[i] > 0 and val >= 0:
                # numerical
                grade[i] = self.get_fuzzy_set(i).get_membership_value(val)
            elif self.__antecedent_indices[i] == 0:
                # don't care
                grade[i] = 1.0
            else:
                raise Exception("Illegal argument")

        return grade

    def get_compatible_grade_value(self, attribute_vector):
        if self.get_antecedent_length() != attribute_vector.get_num_dim():
            raise Exception("antecedent_indices and attribute_vector must have the same length")

        grade = self.get_compatible_grade(attribute_vector)
        return np.prod(grade)

    def get_rule_length(self):
        return np.count_nonzero(self.__antecedent_indices)

    def copy(self):
        return Antecedent(self.__antecedent_indices)

    def get_fuzzy_set(self, dim):
        return Context.get_instance().get_fuzzy_set(dim, self.__antecedent_indices)

    # def get_fuzzy_sets(antecedent_indices):
    # def to_element(self):
