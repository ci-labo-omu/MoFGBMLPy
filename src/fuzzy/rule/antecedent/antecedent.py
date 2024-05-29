import numpy as np
from src.fuzzy.context.context import Context


class Antecedent:
    @staticmethod
    def get_compatible_grade(antecedent_indices, attribute_vector):
        # compute membership value
        grade = np.zeros(len(antecedent_indices))

        if len(antecedent_indices) != attribute_vector.get_num_dim():
            raise Exception("antecedent_indices and attribute_vector must have the same length")

        for i in range(attribute_vector.get_num_dim()):
            val = attribute_vector.get_value(i)
            if antecedent_indices[i] < 0 and val < 0:
                # categorical
                grade[i] = 1.0 if antecedent_indices[i] == round(val) else 0.0
            elif antecedent_indices[i] > 0 and val >= 0:
                # numerical
                grade[i] = Antecedent.get_fuzzy_set(i, antecedent_indices[i]).get_membership_value(val)
            elif antecedent_indices[i] == 0:
                # don't care
                grade[i] = 1.0
            else:
                raise Exception("Illegal argument")

        return grade

    @staticmethod
    def get_compatible_grade_value(antecedent_indices, attribute_vector):
        if len(antecedent_indices) != attribute_vector.get_num_dim():
            raise Exception("antecedent_indices and attribute_vector must have the same length")

        grade = Antecedent.get_compatible_grade(antecedent_indices, attribute_vector)
        return np.prod(grade)

    @staticmethod
    def get_rule_length(antecedent_indices):
        return np.count_nonzero(antecedent_indices)

    @staticmethod
    def copy():
        return Antecedent()

    @staticmethod
    def get_fuzzy_set(dim, antecedent_indices):
        return Context.get_instance().get_fuzzy_set(dim, antecedent_indices)

    # def get_fuzzy_sets(antecedent_indices):
    # def to_element(self):
