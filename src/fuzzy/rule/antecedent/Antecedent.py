import numpy as np

class Antecedent:
    def __init__(self):
        return
        # TODO

    def get_compatible_grade(self, antecedent_indices, attribute_vector):
        # compute membership value
        grade = np.zeros(len(antecedent_indices))

        if len(antecedent_indices) != attribute_vector.get_num_dim()