from main.consts import Consts
from src.fuzzy.rule.antecedent.factory.abstract_antecedent_factory import AbstractAntecedentFactory
from src.fuzzy.knowledge.knowledge import Knowledge
from src.fuzzy.rule.antecedent.antecedent import Antecedent
import numpy as np
import random


class HeuristicAntecedentFactory(AbstractAntecedentFactory):
    __dimension = None
    __training_set = None

    def __init__(self, training_set):
        self.__dimension = Knowledge.get_instance().get_num_dim()
        self.__training_set = training_set

    def select_antecedent_part(self, index):
        pattern = self.__training_set.get_pattern(index)
        return self.calculate_antecedent_part(pattern)

    def calculate_antecedent_part(self, pattern):
        attribute_array = pattern.get_attributes_vector()

        if Consts.IS_DONT_CARE_PROBABILITY:
            dc_rate = Consts.DONT_CARE_RT
        else:
            dc_rate = max((self.__dimension - Consts.ANTECEDENT_NUMBER_NOT_DONT_CARE) / self.__dimension, Consts.DONT_CARE_RT)

        antecedent_indices = np.zeros(self.__dimension, dtype=np.int_)
        knowledge = Knowledge.get_instance()

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
            num_fuzzy_sets_not_dc = knowledge.get_num_fuzzy_sets(dim_i)-1
            if num_fuzzy_sets_not_dc < 1:
                antecedent_indices[dim_i] = 0  # don't care
                continue

            mb_values_inc_sums = np.zeros(num_fuzzy_sets_not_dc, dtype=np.float_)
            sum_mb_values = 0
            for h in range(num_fuzzy_sets_not_dc):
                sum_mb_values += knowledge.get_membership_value(attribute_array[dim_i], dim_i, h+1)
                mb_values_inc_sums[h] = sum_mb_values

            arrow = random.random() * sum_mb_values

            for h in range(num_fuzzy_sets_not_dc):
                if arrow < mb_values_inc_sums[h]:
                    antecedent_indices[dim_i] = h+1
                    break

        return antecedent_indices

    def create(self, num_rules=None):
        indices = self.create_antecedent_indices(num_rules)

        antecedent_objects = np.array([Antecedent(self.select_antecedent_part((indices[i]))) for i in range(num_rules)], dtype=object)

        if num_rules == 1:
            antecedent_objects = antecedent_objects[0]
        return antecedent_objects

    def create_antecedent_indices(self, num_rules=None):
        data_size = self.__training_set.get_size()
        if num_rules is None:
            pattern_index = random.randint(0, data_size - 1)
            return self.select_antecedent_part(pattern_index)

        if num_rules <= self.__training_set.get_size():
            return np.random.choice(list(range(self.__training_set.get_size())), num_rules, replace=False)
        else:
            indices = [i for i in range(data_size)] * (num_rules // data_size)

            num_remaining_indices = num_rules % data_size
            remaining_indices = np.random.choice(list(range(self.__training_set.get_size())), num_remaining_indices,
                                                 replace=False)
            return np.concatenate((indices, remaining_indices))

    def __str__(self):
        return "HeuristicAntecedentFactory [dimension=" + str(self.__dimension) + "]"

    def __copy__(self):
        return HeuristicAntecedentFactory(self.__training_set)
