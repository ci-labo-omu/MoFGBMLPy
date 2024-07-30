import xml.etree.cElementTree as xml_tree
import copy
import time

import numpy as np
cimport numpy as cnp

from mofgbmlpy.data.dataset cimport Dataset
from mofgbmlpy.data.pattern cimport Pattern
from mofgbmlpy.fuzzy.classifier.classifier cimport Classifier
from mofgbmlpy.gbml.solution.abstract_solution cimport AbstractSolution
from mofgbmlpy.fuzzy.knowledge.knowledge import Knowledge
from mofgbmlpy.gbml.solution.abstract_solution cimport AbstractSolution
from mofgbmlpy.gbml.solution.michigan_solution cimport MichiganSolution
from mofgbmlpy.gbml.solution.michigan_solution_builder cimport MichiganSolutionBuilder

cdef class PittsburghSolution(AbstractSolution):
    def __init__(self, num_vars, num_objectives, num_constraints, michigan_solution_builder, classifier, do_init_vars=True):
        super().__init__(num_objectives, num_constraints)
        self.__michigan_solution_builder = michigan_solution_builder
        self.__classifier = classifier
        if do_init_vars:
            self._vars = michigan_solution_builder.create(num_vars)

    cpdef MichiganSolutionBuilder get_michigan_solution_builder(self):
        return self.__michigan_solution_builder


    cdef void learning(self):
        for var in self._vars:
            var.learning()

    cdef AbstractSolution classify(self, Pattern pattern):
        return self.__classifier.classify(self.get_vars(), pattern)

    cpdef double get_error_rate(self, Dataset dataset):
        return self.__classifier.get_error_rate(self.get_vars(), dataset)

    cpdef object[:] get_errored_patterns(self, Dataset dataset):
        return self.__classifier.get_errored_patterns(self.get_vars(), dataset)

    cpdef double compute_coverage(self):
        coverage = 0
        for michigan_solution in self._vars:
            coverage += michigan_solution.compute_coverage()
        return coverage

    cpdef int get_total_rule_length(self):
        return self.__classifier.get_length(self._vars)

    cpdef double get_average_rule_weight(self):
        cdef double total_rule_weight = 0
        cdef int i
        cdef MichiganSolution var

        for i in range(self._vars.shape[0]):
            var = self._vars[i]
            total_rule_weight += var.get_rule_weight_py().get_value()

        return total_rule_weight/self._vars.shape[0]

    def __deepcopy__(self, memo={}):
        new_solution = PittsburghSolution(self.get_num_vars(),
                                          self.get_num_objectives(),
                                          self.get_num_constraints(),
                                          copy.deepcopy(self.__michigan_solution_builder),
                                          copy.deepcopy(self.__classifier),
                                          do_init_vars=False)

        cdef MichiganSolution[:] vars_copy = np.empty(self.get_num_vars(), dtype=object)
        cdef double[:] objectives_copy = np.empty(self.get_num_objectives())
        cdef int i

        for i in range(vars_copy.shape[0]):
            vars_copy[i] = copy.deepcopy(self._vars[i])

        for i in range(objectives_copy.shape[0]):
            objectives_copy[i] = self._objectives[i]

        new_solution._vars = vars_copy
        new_solution._objectives = objectives_copy

        memo[id(self)] = new_solution

        return new_solution

    def __copy__(self):
        return self.__deepcopy__() # pymoo use copy so it causes issues

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

    def __repr__(self):
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