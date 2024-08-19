from mofgbmlpy.data.pattern cimport Pattern
from mofgbmlpy.exception.uninitialized_knowledge_exception import UninitializedKnowledgeException
from mofgbmlpy.fuzzy.rule.antecedent.factory.abstract_antecedent_factory cimport AbstractAntecedentFactory
from mofgbmlpy.fuzzy.knowledge.knowledge import Knowledge
from mofgbmlpy.fuzzy.rule.antecedent.antecedent cimport Antecedent
import numpy as np


cdef class HeuristicAntecedentFactory(AbstractAntecedentFactory):
    """Antecedent factory that don't create antecedent in the constructor and use patterns to generate them in a random way but considering the compatibility grade values
    Note that if the compatibility grade of all the fuzzy sets is null, then even if the dc_rate is set to 0 there might be some DC fuzzy sets. This was the same behavior in the Java version

    Attributes:
        __training_set (Dataset): Training dataset from where patterns are extracted to generate antecedent when no pattern is provided
        __knowledge (Knowledge): Knowledge base
        __is_dc_probability (bool): If True then use dc_rate to determine the number of don't care in the generated antecedent otherwise use antecedent_number_do_not_dont_care to compute the dc_rate
        __dc_rate (double): Between 0 and 1, gives the ratio of don't care compared to not don't care fuzzy sets in the antecedent
        __antecedent_number_do_not_dont_care (int): Number of fuzzy sets that should not be don't care
        _random_gen (numpy.random.Generator): Random generator
    """
    def __init__(self, Dataset training_set, Knowledge knowledge, bint is_dc_probability, double dc_rate, int antecedent_number_do_not_dont_care, random_gen):
        """Constructor

        Args:
            training_set (Dataset): Training dataset from where patterns are extracted to generate antecedent when no pattern is provided
            knowledge (Knowledge): Knowledge base
            is_dc_probability (bool): If True then use dc_rate to determine the number of don't care in the generated antecedent otherwise use antecedent_number_do_not_dont_care to compute the dc_rate
            dc_rate (double): Between 0 and 1, gives the ratio of don't care compared to not don't care fuzzy sets in the antecedent
            antecedent_number_do_not_dont_care (int): Number of fuzzy sets that should not be don't care
            random_gen (numpy.random.Generator): Random generator
        """
        if knowledge is None:
            raise TypeError("Knowledge can't be None")
        elif knowledge.get_num_dim() == 0:
            raise UninitializedKnowledgeException()

        if training_set is None:
            raise TypeError("Training set can't be None")
        elif training_set.get_size() == 0 or knowledge.get_num_dim() != training_set.get_num_dim():
            raise ValueError("Training set must have at least one element and with the same number of dimensions as in the knowledge")

        if dc_rate < 0 or dc_rate > 1:
            raise ValueError("dc rate must be between 0 and 1")

        if antecedent_number_do_not_dont_care < 0:
            raise ValueError(f"antecedent num not dont care must not be negative")


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
        """Select the pattern at the given index in the dataset and use it to compute the antecedent part. Can only be accessed from Cython code
        
        Args:
            index (int): Index of the pattern used to generate the antecedent indices

        Returns:
            int[]: Antecedent indices generated  
        """
        pattern = self.__training_set.get_pattern(index)
        return self.calculate_antecedent_part(pattern)

    cdef int[:] calculate_antecedent_part(self, Pattern pattern):
        """Use a pattern to generate antecedent indices using its compatibility grades with the different fuzzy sets. Can only be accessed from Cython code
        
        Args:
            pattern (Pattern): Pattern used to generate the antecedent indices

        Returns:
            int[]: Antecedent indices generated  
        """
        if pattern is None:
            raise TypeError("The pattern is none")

        cdef double[:] attribute_array = pattern.get_attributes_vector()
        cdef int dim_i
        cdef int h
        cdef int dimension = self.__knowledge.get_num_dim()

        if pattern.get_num_dim() != dimension:
            raise ValueError("Pattern dimension must be the same as the current knowledge")


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
                sum_mb_values += self.__knowledge.get_membership_value(attribute_array[dim_i], dim_i, h+1)
                mb_values_inc_sums[h] = sum_mb_values

            arrow = self._random_gen.random() * sum_mb_values

            for h in range(num_fuzzy_sets_not_dc):
                if arrow < mb_values_inc_sums[h]:
                    antecedent_indices[dim_i] = h+1
                    break

        return antecedent_indices

    cpdef int[:] calculate_antecedent_part_py(self, Pattern pattern):
        """Use a pattern to generate antecedent indices using its compatibility grades with the different fuzzy sets

        Args:
            pattern (Pattern): Pattern used to generate the antecedent indices

        Returns:
            int[]: Antecedent indices generated  
        """
        return self.calculate_antecedent_part(pattern)

    cdef Antecedent[:] create(self, int num_rules=1):
        """Create antecedents. Can only be accessed from Cython code
        
        Args:
            num_rules (int): Number of antecedent to be generated

        Returns:
            Antecedent[]: Generated antecedent
        """
        if num_rules <= 0:
            raise ValueError("num_rules must be positive")
        cdef int[:,:] indices = self.create_antecedent_indices(num_rules)
        cdef int i
        cdef Antecedent[:] antecedent_objects = np.array([Antecedent(indices[i], self.__knowledge) for i in range(num_rules)], dtype=object)

        return antecedent_objects

    cpdef Antecedent[:] create_py(self, int num_rules=1):
        """Create antecedents.

        Args:
            num_rules (int): Number of antecedent to be generated

        Returns:
            Antecedent[]: Generated antecedent
        """
        return self.create(num_rules)

    cdef int[:,:] create_antecedent_indices_from_pattern(self, Pattern pattern):
        """Create an antecedent indices array and return it inside an array. Use a pattern to generate the antecedent indices using its compatibility grades with the different fuzzy sets. Can only be accessed from Cython code

        Args:
            pattern (Pattern): Pattern used to generate the antecedent indices

        Returns:
            int[]: Antecedent indices generated  
        """
        if pattern is None:
            raise TypeError("The pattern is none")
        return np.array([self.calculate_antecedent_part(pattern)], dtype=int)


    cpdef int[:,:] create_antecedent_indices_from_pattern_py(self, Pattern pattern):
        """Create an antecedent indices array and return it inside an array. Use a pattern to generate the antecedent indices using its compatibility grades with the different fuzzy sets

       Args:
           pattern (Pattern): Pattern used to generate the antecedent indices

       Returns:
           int[]: Antecedent indices generated  
       """
        return self.create_antecedent_indices_from_pattern(pattern)

    cdef int[:,:] create_antecedent_indices(self, int num_rules=1):
        """Create antecedents indices. Can only be accessed from Cython code

        Args:
            num_rules (int): Number of antecedents indices arrays to be generated

        Returns:
            int[,]: Generated antecedents indices
        """
        cdef int data_size = self.__training_set.get_size()
        cdef int i
        cdef int j
        cdef int k
        cdef int pattern_index
        cdef int[:] pattern_indices
        cdef int num_remaining_indices
        cdef int[:,:] new_antecedent_indices

        if num_rules <= 0:
            raise ValueError("num_rules must be positive")

        if num_rules is None or num_rules == 1:
            pattern_index = self._random_gen.integers(0, data_size)
            return np.array([self.__select_antecedent_part(pattern_index)], dtype=int)

        if num_rules <= self.__training_set.get_size():
            pattern_indices = self._random_gen.choice(np.arange(self.__training_set.get_size(), dtype=int), num_rules, replace=False)

        else:
            pattern_indices = np.empty(num_rules, int)

            k = 0
            for i in range(num_rules // data_size):
                for j in range(data_size):
                    pattern_indices[k] = j
                    k += 1

            num_remaining_indices = num_rules % data_size
            remaining_indices = self._random_gen.choice(np.arange(self.__training_set.get_size(), dtype=int), num_remaining_indices,
                                                 replace=False)

            for i in range(num_remaining_indices):
                pattern_indices[k] = remaining_indices[i]
                k += 1
        new_antecedent_indices = np.empty((num_rules, self.__knowledge.get_num_dim()), dtype=int)

        for i in range(num_rules):
            new_antecedent_indices[i] = self.__select_antecedent_part(pattern_indices[i])
        return new_antecedent_indices

    cpdef int[:,:] create_antecedent_indices_py(self, int num_rules=1):
        """Create antecedents indices.

        Args:
            num_rules (int): Number of antecedents indices arrays to be generated

        Returns:
            int[,]: Generated antecedents indices
        """
        return self.create_antecedent_indices(num_rules)

    def __repr__(self):
        """Return a string representation of this object

        Returns:
            (str) String representation
        """
        return "HeuristicAntecedentFactory [dimension=" + str(self.__knowledge.get_num_dim()) + "]"

    def __eq__(self, other):
        """Check if another object is equal to this one

        Args:
            other (object): Object compared to this one

        Returns:
            bool: True if they are equal and False otherwise
        """
        if not isinstance(other, HeuristicAntecedentFactory):
            return False

        return (self.__training_set == other.get_training_set() and
                self.__knowledge == other.get_knowledge() and
                self.__is_dc_probability == other.get_is_dc_probability() and
                self.__dc_rate == other.get_dc_rate() and
                self.__antecedent_number_do_not_dont_care == other.get_antecedent_number_do_not_dont_care() and
                self._random_gen == other.get_random_gen())

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

    cpdef get_training_set(self):
        """Get the training set

        Returns:
            Dataset: The training set
        """
        return self.__training_set

    cpdef get_knowledge(self):
        """Get the knowledge base

        Returns:
            Knowledge: The knowledge base
        """
        return self.__knowledge

    cpdef get_is_dc_probability(self):
        """Get the is_dc_probability

        Returns:
            bool: is_dc_probability
        """
        return self.__is_dc_probability

    cpdef get_dc_rate(self):
        """Get the dc_rate

        Returns:
            double: dc_rate
        """
        return self.__dc_rate

    cpdef get_antecedent_number_do_not_dont_care(self):
        """Get the antecedent_number_do_not_dont_care

        Returns:
            int: antecedent_number_do_not_dont_care
        """
        return self.__antecedent_number_do_not_dont_care

    cpdef get_random_gen(self):
        """Get the random generator

        Returns:
            numpy.random.Generator: The random generator instance.
        """
        return self._random_gen