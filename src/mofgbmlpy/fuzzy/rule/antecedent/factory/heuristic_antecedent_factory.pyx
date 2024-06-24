from mofgbmlpy.fuzzy.rule.antecedent.factory.abstract_antecedent_factory import AbstractAntecedentFactory
from mofgbmlpy.fuzzy.knowledge.knowledge import Knowledge
from mofgbmlpy.fuzzy.rule.antecedent.antecedent import Antecedent
import numpy as np
import random


class HeuristicAntecedentFactory(AbstractAntecedentFactory):
    __dimension = None
    __training_set = None
    __knowledge = None
    __is_dc_probability = None
    __dc_rate = None
    __antecedent_num_not_dont_care = None

    def __init__(self, training_set, knowledge, is_dc_probability, dc_rate, antecedent_num_not_dont_care):
        self.__dimension = knowledge.get_num_dim()
        self.__training_set = training_set
        self.__knowledge = knowledge
        self.__is_dc_probability = is_dc_probability
        self.__dc_rate = dc_rate
        self.__antecedent_num_not_dont_care = antecedent_num_not_dont_care

    def select_antecedent_part(self, index):
        pattern = self.__training_set.get_pattern(index)
        return self.calculate_antecedent_part(pattern)

    def calculate_antecedent_part(self, pattern):
        attribute_array = pattern.get_attributes_vector()

        if self.__is_dc_probability:
            dc_rate = self.__dc_rate
        else:
            dc_rate = max((self.__dimension - self.__antecedent_num_not_dont_care) / self.__dimension, self.__dc_rate)

        antecedent_indices = np.zeros(self.__dimension, dtype=np.int_)

        for dim_i in range(self.__dimension):
            # DC
            if random.random() < dc_rate:
                antecedent_indices[dim_i] = 0  # The first fuzzy set is don't care
                continue

            # Categorical judge
            # TODO: Check back later if it's used in the code
            if attribute_array[dim_i] < 0:
                antecedent_indices[dim_i] = int(attribute_array[dim_i])
                continue

            # Numerical (get a random fuzzy set index using the membership value)
            num_fuzzy_sets_not_dc = self.__knowledge.get_num_fuzzy_sets(dim_i)-1
            if num_fuzzy_sets_not_dc < 1:
                antecedent_indices[dim_i] = 0  # don't care
                continue

            mb_values_inc_sums = np.zeros(num_fuzzy_sets_not_dc, dtype=np.float64)
            sum_mb_values = 0
            for h in range(num_fuzzy_sets_not_dc):
                sum_mb_values += self.__knowledge.get_membership_value(attribute_array[dim_i], dim_i, h+1)
                mb_values_inc_sums[h] = sum_mb_values

            arrow = random.random() * sum_mb_values

            for h in range(num_fuzzy_sets_not_dc):
                if arrow < mb_values_inc_sums[h]:
                    antecedent_indices[dim_i] = h+1
                    break

        return antecedent_indices

    def create(self, num_rules=None):
        indices = self.create_antecedent_indices(num_rules)
        antecedent_objects = np.array([Antecedent(self.select_antecedent_part((indices[i])), self.__knowledge) for i in range(num_rules)], dtype=object)

        return antecedent_objects

    def create_antecedent_indices_from_pattern(self, pattern=None):
        if pattern is None:
            raise Exception("Pattern cannot be None")
        return np.array([self.calculate_antecedent_part(pattern)], dtype=int)

    def create_antecedent_indices(self, num_rules=None):
        data_size = self.__training_set.get_size()
        if num_rules is None:
            pattern_index = random.randint(0, data_size - 1)
            return np.array([self.select_antecedent_part(pattern_index)], dtype=int)

        if num_rules <= self.__training_set.get_size():
            pattern_indices = np.random.choice(list(range(self.__training_set.get_size())), num_rules, replace=False)
            # print(result.shape, result)
            if len(pattern_indices.shape) == 1:
                pattern_indices = np.array([pattern_indices], dtype=int)

        else:
            # TODO: this function seems invalid (indices is maybe an array of copies and the concatenation is maybe between incompatible types (int and array)
            pattern_indices = [i for i in range(data_size)] * (num_rules // data_size)

            num_remaining_indices = num_rules % data_size
            remaining_indices = np.random.choice(list(range(self.__training_set.get_size())), num_remaining_indices,
                                                 replace=False)
            pattern_indices = np.concatenate((pattern_indices, remaining_indices))

        new_antecedent_indices = np.empty((num_rules, self.__dimension), dtype=int)
        for i in range(num_rules):
            new_antecedent_indices[i] = self.select_antecedent_part(pattern_indices[i])
        return new_antecedent_indices

    def __str__(self):
        return "HeuristicAntecedentFactory [dimension=" + str(self.__dimension) + "]"

    def __copy__(self):
        return HeuristicAntecedentFactory(self.__training_set, self.__knowledge, self.__is_dc_probability, self.__dc_rate, self.__antecedent_num_not_dont_care)
