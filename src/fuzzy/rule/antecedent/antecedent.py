import numpy as np
from src.fuzzy.knowledge.knowledge import Knowledge


class Antecedent:
    __antecedent_indices = None
    __knowledge = None

    def __init__(self, antecedent_indices, knowledge):
        self.__antecedent_indices = antecedent_indices
        self.__knowledge = knowledge

    def get_array_size(self):
        return len(self.__antecedent_indices)

    def get_antecedent_indices(self):
        return self.__antecedent_indices

    def set_antecedent_indices(self, new_indices):
        self.__antecedent_indices = new_indices

    def get_compatible_grade(self, attribute_vector):
        # compute membership value
        grade = np.zeros(self.get_array_size())

        if self.get_array_size() != len(attribute_vector):
            raise Exception("antecedent_indices and attribute_vector must have the same length")

        for i in range(len(attribute_vector)):
            val = attribute_vector[i]
            if self.__antecedent_indices[i] < 0 and val < 0:
                # categorical
                grade[i] = 1.0 if self.__antecedent_indices[i] == round(val) else 0.0
            elif self.__antecedent_indices[i] > 0 and val >= 0:
                # numerical
                grade[i] = self.__knowledge.get_membership_value(val, i, self.__antecedent_indices[i])
            elif self.__antecedent_indices[i] == 0:
                # don't care
                grade[i] = 1.0
            else:
                raise Exception("Illegal argument")

        return grade

    def get_compatible_grade_value(self, attribute_vector):
        grade = self.get_compatible_grade(attribute_vector)
        return np.prod(grade)

    def get_length(self):
        return np.count_nonzero(self.__antecedent_indices)

    def __copy__(self):
        return Antecedent(self.__antecedent_indices, knowledge=self.__knowledge)
