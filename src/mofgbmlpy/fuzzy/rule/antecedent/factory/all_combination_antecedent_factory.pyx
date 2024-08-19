# distutils: language = c++

import copy
cimport numpy as cnp
from libcpp.queue cimport queue as cqueue
from libcpp.vector cimport vector as cvector
from libc cimport math as cmath

from mofgbmlpy.exception.uninitialized_knowledge_exception import UninitializedKnowledgeException
from mofgbmlpy.fuzzy.fuzzy_term.fuzzy_variable cimport FuzzyVariable
from mofgbmlpy.fuzzy.knowledge.knowledge cimport Knowledge
from mofgbmlpy.fuzzy.rule.antecedent.factory.abstract_antecedent_factory cimport AbstractAntecedentFactory
from mofgbmlpy.fuzzy.rule.antecedent.antecedent cimport Antecedent
import numpy as np

cdef extern from "limits.h":
    cdef int INT_MAX

cdef class AllCombinationAntecedentFactory(AbstractAntecedentFactory):
    """Antecedent factory creating all the possible antecedents in the constructor

    Attributes:
        __antecedents_indices (int[,]): All the possible antecedent indices for the current knowledge base
        __knowledge (Knowledge): Knowledge base
        _random_gen (numpy.random.Generator): Random generator
    """
    def __init__(self, knowledge, random_gen):
        """Constructor

        Args:
            knowledge (Knowledge): Knowledge base
            random_gen (numpy.random.Generator): Random generator
        """
        if knowledge is None:
            raise TypeError("knowledge can't be None")
        elif knowledge.get_num_dim() == 0:
            raise UninitializedKnowledgeException()

        self.__knowledge = knowledge
        self.__antecedents_indices = self.generate_antecedents_indices()
        self._random_gen = random_gen

    def get_num_antecedents(self):
        """Get the number of generated antecedent indices arrays

        Returns:
            int: Number of generated antecedent indices arrays
        """
        return len(self.__antecedents_indices)

    cdef int[:,:] generate_antecedents_indices(self):
        """Generate all the possible combinations of fuzzy sets indices from the knowledge base. Can only be accessed from Cython code
        
        Returns:
            int[,]: All the combinations of fuzzy sets indices (it's an array of antecedent indices arrays)
        
        Raises:
            Exception: The number of combinations is too big
        """
        cdef int i
        cdef int j
        cdef int k = 0
        cdef int current_dim
        cdef cqueue[cvector[int]] indices_queue
        cdef int num_generated_indices = 1
        cdef cvector[int] tmp
        cdef FuzzyVariable var
        cdef int dimension = self.__knowledge.get_num_dim()
        cdef FuzzyVariable[:] fuzzy_vars = self.__knowledge.get_fuzzy_vars()
        cdef int var_length

        for i in range(dimension):
            var = fuzzy_vars[i]
            var_length = var.get_length()
            if num_generated_indices > INT_MAX / var_length:
                print("WARNING: Too many antecedent indices to be generated, not all combinations will be generated")
                num_generated_indices = INT_MAX
                break
            num_generated_indices *= var_length

        cdef int[:,:] indices

        try:
            indices = (np.empty((num_generated_indices, dimension), dtype=int))
        except MemoryError:
            raise MemoryError("The number of variables and/or the number of fuzzy sets is too big,"
                            " the antecedents list memory can't be allocated."
                            "Please use another antecedent factory")

        indices_queue.push(cvector[int]())

        # Generate all combination of fuzzy sets indices
        while indices_queue.size() > 0:
            buffer = indices_queue.front()
            indices_queue.pop()
            current_dim = buffer.size()
            if current_dim < dimension:
                var = fuzzy_vars[current_dim]
                for i in range(var.get_length()):
                    tmp = cvector[int]()

                    for j in range(current_dim):
                        tmp.push_back(buffer[j])
                    tmp.push_back(i)
                    indices_queue.push(tmp)
            else:
                # A list of antecedent indices is full so we can add it
                for i in range(dimension):
                    indices[k][i] = buffer[i]
                k += 1
                if k >= num_generated_indices:
                    break

        return indices

    def generate_antecedents_indices_py(self):
        """Generate all the possible combinations of fuzzy sets indices from the knowledge base.

        Returns:
            int[,]: All the combinations of fuzzy sets indices (it's an array of antecedent indices arrays)

        Raises:
            Exception: The number of combinations is too big
        """
        return self.generate_antecedents_indices()

    cdef Antecedent[:] create(self, int num_rules=1):
        """Create antecedents. Can only be accessed from Cython code
        
        Args:
            num_rules (int): Number of antecedent to be generated

        Returns:
            Antecedent[]: Generated antecedent
        """
        cdef int[:,:] indices = self.create_antecedent_indices(num_rules)
        cdef Antecedent[:] antecedent_objects = np.zeros(num_rules, dtype=object)
        cdef int i
        cdef Antecedent new_antecedent_obj

        for i in range(num_rules):
            new_antecedent_obj = Antecedent(indices[i], self.__knowledge)
            antecedent_objects[i] = new_antecedent_obj

        return antecedent_objects

    def create_py(self, int num_rules=1):
        """Create antecedents.

        Args:
            num_rules (int): Number of antecedent to be generated

        Returns:
            Antecedent[]: Generated antecedent
        """
        return self.create(num_rules)


    cdef int[:,:] create_antecedent_indices(self, int num_rules=1):
        """Create antecedents indices. Can only be accessed from Cython code

        Args:
            num_rules (int): Number of antecedents indices arrays to be generated

        Returns:
            int[,]: Generated antecedents indices
        """
        cdef int i
        cdef int j
        cdef int[:] chosen_list

        if num_rules <= 0:
            raise ValueError("num_rules must be positive")

        num_rules = min(num_rules, self.__antecedents_indices.shape[0])

        # Return an antecedent
        cdef int[:] chosen_indices_lists = self._random_gen.choice(np.arange(self.__antecedents_indices.shape[0], dtype=int), num_rules, replace=False)
        cdef int[:,:] new_indices = np.empty((num_rules, self.__knowledge.get_num_dim()), dtype=int)

        for i in range(chosen_indices_lists.shape[0]):
            chosen_list = self.__antecedents_indices[chosen_indices_lists[i]]
            for j in range(chosen_list.shape[0]):
                new_indices[i][j] = chosen_list[j]
        return new_indices

    def create_antecedent_indices_py(self, int num_rules=1):
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
        return "AllCombinationAntecedentFactory [antecedents=" + str(self.__antecedents_indices) + ", dimension=" + str(
            self.__knowledge.get_num_dim()) + "]"

    def __eq__(self, other):
        """Check if another object is equal to this one
        
        Args:
            other (object): Object compared to this one 

        Returns:
            (bool) True if they are equal and False otherwise
        """
        if not isinstance(other, AllCombinationAntecedentFactory):
            return False

        return self.__knowledge == other.get_knowledge()

    def __deepcopy__(self, memo={}):
        """Return a deepcopy of this object

        Args:
            memo (dict): Dictionary of objects already copied during the current copying pass;

        Returns:
            object: Deep copy of this object
        """
        new_object = AllCombinationAntecedentFactory(self.__knowledge, self._random_gen)

        memo[id(self)] = new_object
        return new_object

    cpdef Knowledge get_knowledge(self):
        """Get the knowledge base used to create the antecedents
        
        Returns:
            Knowledge: Knowledge base
        """
        return self.__knowledge