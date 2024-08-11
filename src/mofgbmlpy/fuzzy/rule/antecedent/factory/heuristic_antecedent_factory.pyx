from mofgbmlpy.data.pattern cimport Pattern
from mofgbmlpy.fuzzy.rule.antecedent.factory.abstract_antecedent_factory cimport AbstractAntecedentFactory
from mofgbmlpy.fuzzy.knowledge.knowledge import Knowledge
from mofgbmlpy.fuzzy.rule.antecedent.antecedent cimport Antecedent
import numpy as np


cdef class HeuristicAntecedentFactory(AbstractAntecedentFactory):
    def __init__(self, Dataset training_set, Knowledge knowledge, bint is_dc_probability, double dc_rate, int antecedent_number_do_not_dont_care, random_gen):
        if knowledge is None or knowledge.get_num_dim() == 0:
            raise Exception("knowledge can't be None and must have at least one fuzzy variable")

        if training_set is None or training_set.get_size() == 0 or knowledge.get_num_dim() != training_set.get_num_dim():
            raise Exception("training set must have at least one element and with the same number of dimensions as in the knowledge")

        if dc_rate < 0 or dc_rate > 1:
            raise Exception("dc rate must be between 0 and 1")

        if antecedent_number_do_not_dont_care < 0:
            raise Exception(f"antecedent num not dont care must not be negative")


        self.__training_set = training_set
        self.__knowledge = knowledge
        self.__is_dc_probability = is_dc_probability
        self.__antecedent_number_do_not_dont_care = antecedent_number_do_not_dont_care
        self._random_gen = random_gen

        if self.__is_dc_probability:
            self.__dc_rate = dc_rate
        else:
            self.__dc_rate = max((self.__knowledge.get_num_dim() - self.__antecedent_number_do_not_dont_care) / self.__knowledge.get_num_dim(), dc_rate)

    cdef int[:] __select_antecedent_part(self, int index):
        pattern = self.__training_set.get_pattern(index)
        return self.calculate_antecedent_part(pattern)

    cdef int[:] calculate_antecedent_part(self, Pattern pattern):
        if pattern is None:
            raise Exception("Pattern can't be none")

        cdef double[:] attribute_array = pattern.get_attributes_vector()
        cdef int dim_i
        cdef int h
        cdef int dimension = self.__knowledge.get_num_dim()

        if pattern.get_num_dim() != dimension:
            raise Exception("Pattern dimension must be the same as the current knowledge")


        antecedent_indices = np.zeros(dimension, dtype=np.int_)

        for dim_i in range(dimension):
            # DC
            if self._random_gen.random() < self.__dc_rate:
                antecedent_indices[dim_i] = 0  # The first fuzzy set (index = 0) is don't care
                continue

            # Categorical judge
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
                sum_mb_values += self.__knowledge.get_membership_value_py(attribute_array[dim_i], dim_i, h+1)
                mb_values_inc_sums[h] = sum_mb_values

            arrow = self._random_gen.random() * sum_mb_values

            for h in range(num_fuzzy_sets_not_dc):
                if arrow < mb_values_inc_sums[h]:
                    antecedent_indices[dim_i] = h+1
                    break

        return antecedent_indices

    def calculate_antecedent_part_py(self, Pattern pattern):
        return self.calculate_antecedent_part(pattern)

    cdef Antecedent[:] create(self, int num_rules=1):
        if num_rules <= 0:
            raise Exception("num_rules must be positive")
        cdef int[:,:] indices = self.create_antecedent_indices(num_rules)
        cdef int i
        cdef Antecedent[:] antecedent_objects = np.array([Antecedent(indices[i], self.__knowledge) for i in range(num_rules)], dtype=object)

        return antecedent_objects

    cdef int[:,:] create_antecedent_indices_from_pattern(self, Pattern pattern=None):
        if pattern is None:
            raise Exception("Pattern cannot be None")
        return np.array([self.calculate_antecedent_part(pattern)], dtype=int)

    cdef int[:,:] create_antecedent_indices(self, int num_rules=1):
        cdef int data_size = self.__training_set.get_size()
        cdef int i
        cdef int pattern_index
        cdef int[:] pattern_indices
        cdef int num_remaining_indices
        cdef int[:,:] new_antecedent_indices

        if num_rules is None or num_rules == 1:
            pattern_index = self._random_gen.integers(0, data_size)
            return np.array([self.__select_antecedent_part(pattern_index)], dtype=int)

        if num_rules <= self.__training_set.get_size():
            pattern_indices = self._random_gen.choice(np.arange(self.__training_set.get_size(), dtype=int), num_rules, replace=False)

        else:
            # TODO: this function seems invalid (indices is maybe an array of copies and the concatenation is maybe between incompatible types (int and array)
            pattern_indices = [i for i in range(data_size)] * (num_rules // data_size)

            num_remaining_indices = num_rules % data_size
            remaining_indices = self._random_gen.choice(np.arange(self.__training_set.get_size(), dtype=int), num_remaining_indices,
                                                 replace=False)
            pattern_indices = np.concatenate((pattern_indices, remaining_indices))

        new_antecedent_indices = np.empty((num_rules, self.__knowledge.get_num_dim()), dtype=int)
        for i in range(num_rules):
            new_antecedent_indices[i] = self.__select_antecedent_part(pattern_indices[i])
        return new_antecedent_indices

    def __repr__(self):
        """Return a string representation of this object

        Returns:
            (str) String representation
        """
        return "HeuristicAntecedentFactory [dimension=" + str(self.__knowledge.get_num_dim()) + "]"

    def __deepcopy__(self, memo={}):
        """Return a deepcopy of this object

        Args:
            memo (dict): Dictionary of objects already copied during the current copying pass;

        Returns:
            object: Deep copy of this object
        """
        new_object = HeuristicAntecedentFactory(self.__training_set, self.__knowledge, self.__is_dc_probability, self.__dc_rate, self.__antecedent_number_do_not_dont_care, self._random_gen)

        memo[id(self)] = new_object
        return new_object
