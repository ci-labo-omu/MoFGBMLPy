# distutils: language = c++
from libcpp.vector cimport vector as cvector

import xml.etree.cElementTree as xml_tree
import copy
import time

import numpy as np
cimport numpy as cnp

from mofgbmlpy.data.dataset cimport Dataset
from mofgbmlpy.data.pattern cimport Pattern
from mofgbmlpy.fuzzy.classification.abstract_classification import AbstractClassification
from mofgbmlpy.fuzzy.rule.consequent.ruleWeight.rule_weight_multi import RuleWeightMulti
from mofgbmlpy.gbml.solution.abstract_solution cimport AbstractSolution
from mofgbmlpy.fuzzy.knowledge.knowledge import Knowledge
from mofgbmlpy.gbml.solution.abstract_solution cimport AbstractSolution
from mofgbmlpy.gbml.solution.michigan_solution cimport MichiganSolution
from mofgbmlpy.gbml.solution.michigan_solution_builder cimport MichiganSolutionBuilder

cdef class PittsburghSolution(AbstractSolution):
    def __init__(self, num_vars, num_objectives, num_constraints, michigan_solution_builder, classification, do_init_vars=True):
        super().__init__(num_objectives, num_constraints)
        self.__michigan_solution_builder = michigan_solution_builder
        self.__classification = classification
        if do_init_vars:
            self._vars = michigan_solution_builder.create(num_vars)

    cpdef MichiganSolutionBuilder get_michigan_solution_builder(self):
        return self.__michigan_solution_builder


    cpdef void learning(self):
        for var in self._vars:
            var.learning()

    cpdef double compute_coverage(self):
        coverage = 0
        for michigan_solution in self._vars:
            coverage += michigan_solution.compute_coverage()
        return coverage

    cpdef double get_average_rule_weight(self):
        cdef double total_rule_weight = 0
        cdef int i
        cdef MichiganSolution var

        for i in range(self._vars.shape[0]):
            var = self._vars[i]
            if isinstance(var.get_rule_weight(), RuleWeightMulti):
                total_rule_weight += np.mean(var.get_rule_weight().get_value())
            else:
                total_rule_weight += var.get_rule_weight().get_value()

        return total_rule_weight/self._vars.shape[0]

    def __deepcopy__(self, memo={}):
        """Return a deepcopy of this object

        Args:
            memo (dict): Dictionary of objects already copied during the current copying pass;

        Returns:
            object: Deep copy of this object
        """
        new_solution = PittsburghSolution(self.get_num_vars(),
                                          self.get_num_objectives(),
                                          self.get_num_constraints(),
                                          copy.deepcopy(self.__michigan_solution_builder),
                                          copy.deepcopy(self.__classification),
                                          do_init_vars=False)

        cdef MichiganSolution[:] vars_copy = np.empty(self.get_num_vars(), dtype=object)
        cdef double[:] objectives_copy = np.empty(self.get_num_objectives())
        cdef int i
        cdef MichiganSolution var

        for i in range(vars_copy.shape[0]):
            var = copy.deepcopy(self._vars[i])
            vars_copy[i] = var

        for i in range(objectives_copy.shape[0]):
            objectives_copy[i] = self._objectives[i]

        new_solution._vars = vars_copy
        new_solution._objectives = objectives_copy

        memo[id(self)] = new_solution

        return new_solution


    def __hash__(self):
        cdef int i
        cdef int hash_val = 13

        for i in range(len(self._vars)):
            hash_val += hash_val * 17 + hash(self._vars[i]) * 17
        return hash_val

    cpdef void remove_var(self, int index):
        self._vars = np.delete(self._vars, index)

    cpdef void clear_vars(self):
        self._vars = np.empty(0, dtype=object)

    cpdef MichiganSolution[:] get_vars(self):
        return self._vars

    cpdef MichiganSolution get_var(self, int index):
        return self._vars[index]

    cpdef void set_var(self, int index, MichiganSolution value):
        self._vars[index] = value

    cpdef void set_vars(self, MichiganSolution[:] new_vars):
        self._vars = new_vars

    cpdef int get_num_vars(self):
        return self._vars.shape[0]

    def __eq__(self, other):
        """Check if another object is equal to this one

        Args:
            other (object): Object compared to this one

        Returns:
            (bool) True if they are equal and False otherwise
        """
        if not isinstance(other, PittsburghSolution):
            return False

        cdef int i
        cdef MichiganSolution[:] other_vars = other.get_vars()
        if self._vars.shape[0] != other_vars.shape[0]:
            return False

        for i in range(len(self._vars)):
            if self._vars[i] != other_vars[i]:
                return False
        return True

    def __repr__(self):
        """Return a string representation of this object

        Returns:
            (str) String representation
        """
        txt = "(Pittsburgh Solution) Variables: ["
        for i in range(self.get_num_vars()):
            txt += f"{self._vars[i]} "

        txt += "] Objectives "
        for i in range(self.get_num_objectives()):
            txt += f"{self._objectives[i]} "

        # txt += "] Constraints "
        # for i in range(self.get_num_constraints()):
        #     txt += f"{self._objectives[i]} "

        txt += f"] Attributes: {self._attributes}"

        return txt

    def to_xml(self):
        """Get the XML representation of this object.

        Returns:
            (xml.etree.ElementTree) XML element representing this object
        """
        root = xml_tree.Element("pittsburghSolution")
        for sol in self._vars:
            root.append(sol.to_xml())

        objectives = xml_tree.SubElement(root, "objectives")
        for i in range(self.get_num_objectives()):
            objective = xml_tree.SubElement(objectives, "objective")
            objective.set("id", str(i))
            objective.set("objectiveName", "UNKNOWN")
            objective.text = str(self.get_objective(i))

        attributes = xml_tree.SubElement(root, "attributes")
        for key, value in self.get_attributes().items():
            attribute = xml_tree.SubElement(attributes, "attribute")
            attribute.set("attributeID", key)
            attribute.text = str(value)

        return root

    cpdef bint are_rules_valid(self):
        if self.get_num_vars() < 1:
            return False

        cdef int i
        cdef MichiganSolution sol
        for i in range(self.get_num_vars()):
            sol = self._vars[i]
            if sol.get_rule().is_rejected_class_label():
                return False
        return True

    cdef MichiganSolution classify(self, Pattern pattern):
       return self.__classification.classify(self._vars, pattern)

    cpdef MichiganSolution classify_py(self, Pattern pattern):
       return self.classify(pattern)

    cpdef get_total_rule_length(self):
       length = 0
       if self._vars is not None:
           for item in self._vars:
               length += item.get_length()
       return length

    cpdef double get_error_rate(self, Dataset dataset):
       if self._vars is None or dataset is None:
           raise Exception("Michigan solutions list and dataset can't be None")

       cdef int num_errors = 0
       cdef int dataset_size = dataset.get_size()
       cdef int i
       cdef MichiganSolution winner_solution
       cdef Pattern[:] patterns = dataset.get_patterns()
       cdef Pattern p

       for sol in self._vars:
           sol.reset_num_wins()
           sol.reset_fitness()

       for i in range(dataset.get_size()):
           p = patterns[i]
           winner_solution = self.classify(p)
           if winner_solution is None:
               num_errors += 1
               continue

           winner_solution.inc_num_wins()

           if p.get_target_class() != winner_solution.get_class_label():
               num_errors += 1
           else:
               winner_solution.inc_fitness()

       return num_errors / dataset_size


    cpdef object[:] get_errored_patterns(self, Dataset dataset):
       if self._vars is None or dataset is None:
           raise Exception("Michigan solutions list and dataset can't be None")

       cdef int i
       cdef cvector[int] errored_patterns_indices
       cdef object[:] errored_patterns
       cdef MichiganSolution winner_solution
       cdef Pattern[:] patterns = dataset.get_patterns()
       cdef Pattern p

       for i in range(dataset.get_size()):
           p = patterns[i]
           winner_solution = self.classify(p)

           if winner_solution is None or p.get_target_class() != winner_solution.get_class_label():
               errored_patterns_indices.push_back(i)

       errored_patterns = np.empty(errored_patterns_indices.size(), dtype=object)
       for i in range(errored_patterns_indices.size()):
           errored_patterns[i] = patterns[errored_patterns_indices[i]]

       return errored_patterns


    cpdef AbstractClassification get_classification(self):
       return self.__classification
