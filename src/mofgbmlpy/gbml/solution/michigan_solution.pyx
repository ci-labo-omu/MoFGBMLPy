import xml.etree.cElementTree as xml_tree
import copy

import numpy as np

from mofgbmlpy.data.pattern cimport Pattern
from mofgbmlpy.fuzzy.knowledge.knowledge cimport Knowledge
from mofgbmlpy.fuzzy.rule.rule_builder_core cimport RuleBuilderCore
from mofgbmlpy.fuzzy.rule.abstract_rule cimport AbstractRule
from mofgbmlpy.fuzzy.rule.antecedent.antecedent cimport Antecedent
from mofgbmlpy.fuzzy.rule.consequent.consequent cimport Consequent
from mofgbmlpy.fuzzy.rule.consequent.ruleWeight.abstract_rule_weight cimport AbstractRuleWeight
from mofgbmlpy.gbml.solution.abstract_solution cimport AbstractSolution
cimport numpy as cnp


cdef class MichiganSolution(AbstractSolution):
    def __init__(self, random_gen, num_objectives, num_constraints, rule_builder, pattern=None, do_init_vars=True):
        self._rule_builder = rule_builder
        self.__num_wins = 0
        self.__fitness = 0
        self._random_gen = random_gen

        super().__init__(num_objectives, num_constraints)

        if do_init_vars:
            cnt = 0
            is_rejected = True
            if pattern is None:
                while is_rejected:
                    cnt += 1
                    self.create_rule()
                    is_rejected = self._rule.is_rejected_class_label()
                    if cnt > 1000:
                        raise Exception("Exceeded maximum number of trials to generate rule")
            else:
                training_data_set = self.get_rule_builder().get_training_dataset()
                size_training_data_set = training_data_set.get_size()
                while is_rejected:
                    cnt += 1
                    self.create_rule(training_data_set.get_pattern(random_gen.integers(0, size_training_data_set)))
                    is_rejected = self._rule.is_rejected_class_label()
                    if cnt > 1000:
                        raise Exception("Exceeded maximum number of trials to generate rule")

    cdef void create_rule(self, Pattern pattern=None):
        cdef int[:] antecedent_indices
        antecedent_indices = self._rule_builder.create_antecedent_indices(num_rules=1, pattern=pattern)[0]
        self.set_vars(antecedent_indices)
        self.learning()

    cpdef void learning(self):
        cdef Antecedent antecedent_object
        if self._vars is None:
            raise Exception("Vars is not defined")

        if self._rule is None or self._rule.get_antecedent() is None:
            antecedent_object = self._rule_builder.create_antecedent_from_indices(self._vars)
            self._rule = self._rule_builder.create(antecedent_object)
        else:
            antecedent_object = self._rule.get_antecedent()
            antecedent_object.set_antecedent_indices(self._vars)
            self._rule.set_consequent(self._rule_builder.create_consequent(antecedent_object))

    cpdef double get_fitness_value(self, double[:] in_vector):
        return self._rule.get_fitness_value(in_vector)

    cpdef int get_length(self):
        return self._rule.get_length()

    cpdef get_class_label(self):
        return self._rule.get_class_label()

    cdef AbstractRuleWeight get_rule_weight(self):
        return self._rule.get_rule_weight()

    cpdef AbstractRuleWeight get_rule_weight_py(self):
        return self.get_rule_weight()

    cpdef AbstractRule get_rule(self):
        return self._rule

    cpdef RuleBuilderCore get_rule_builder(self):
        return self._rule_builder

    cpdef Consequent get_consequent(self):
        return self._rule.get_consequent()

    cpdef Antecedent get_antecedent(self):
        return self._rule.get_antecedent()

    cdef double[:] get_compatible_grade(self, double[:] attribute_vector):
        return self._rule.get_compatible_grade(attribute_vector)

    cdef double get_compatible_grade_value(self, double[:] attribute_vector):
        return self._rule.get_compatible_grade_value(attribute_vector)

    cpdef double compute_coverage(self):
        coverage = 1
        dim_i = 0
        for fuzzy_set_index in self._vars:
            coverage *= self._rule_builder.get_knowledge().get_support(dim_i, fuzzy_set_index)
            dim_i += 1
        return coverage

    cpdef void reset_num_wins(self):
        self.__num_wins = 0

    cpdef void reset_fitness(self):
        self.__fitness = 0

    cpdef void inc_num_wins(self):
        self.__num_wins += 1

    cpdef void inc_fitness(self):
        self.__fitness += 1

    cpdef int get_num_wins(self):
        return self.__num_wins

    cpdef int get_fitness(self):
        return self.__fitness

    def __repr__(self):
        """Return a string representation of this object

        Returns:
            (str) String representation
        """
        txt = "(MichiganSolution) Variables: ["
        cdef int i
        for i in range(self.get_num_vars()):
            txt = f"{txt}{self._vars[i]} "

        txt = f"{txt}], Rule weight: {self._rule.get_rule_weight()}, Class label: {self._rule.get_class_label()}"

        txt = f"{txt}], Objectives: ["
        for i in range(self.get_num_objectives()):
            txt = f"{txt}{self._objectives[i]} "

        txt = f"{txt}], Attributes: {{Number of classifier patterns: {self.__fitness}, Number of wins: {self.__num_wins}, "
        for key, val in self._attributes.items():
            txt = f"{txt}{key}: {val}, "
        txt = f"{txt}}}"
        return txt

    def __deepcopy__(self, memo={}):
        """Return a deepcopy of this object

        Args:
            memo (dict): Dictionary of objects already copied during the current copying pass;

        Returns:
            object: Deep copy of this object
        """
        cdef int i
        new_solution = MichiganSolution(self._random_gen,
                                        self.get_num_objectives(),
                                        self.get_num_constraints(),
                                        copy.deepcopy(self._rule_builder),
                                        do_init_vars=False)

        cdef int[:] vars_copy = np.empty(self.get_num_vars(), dtype=int)
        cdef double[:] objectives_copy = np.empty(self.get_num_objectives())


        for i in range(vars_copy.shape[0]):
            vars_copy[i] = self._vars[i]

        for i in range(objectives_copy.shape[0]):
            objectives_copy[i] = self._objectives[i]

        new_solution._rule = copy.deepcopy(self._rule)
        new_solution.__num_wins = self.__num_wins
        new_solution.__fitness = self.__fitness
        new_solution._vars = vars_copy
        new_solution._rule.get_antecedent().set_antecedent_indices(vars_copy)
        new_solution._objectives = objectives_copy

        memo[id(self)] = new_solution

        return new_solution


    def __hash__(self):
        cdef int i
        cdef int hash_val = 13

        for i in range(len(self._vars)):
            hash_val += hash_val * 17 + self._vars[i]
        return hash_val

    cdef void clear_vars(self):
        self._vars = np.empty(0, dtype=int)

    cpdef int[:] get_vars(self):
        return self._vars

    cpdef int get_var(self, int index):
        return self._vars[index]

    cpdef void set_var(self, int index, int value):
        self._vars[index] = value

    cpdef void set_vars(self, int[:] new_vars):
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
        if not isinstance(other, MichiganSolution):
            return False

        cdef int i
        cdef int[:] other_vars = other.get_vars()
        if self._vars.shape[0] != other_vars.shape[0]:
            return False

        for i in range(len(self._vars)):
            if self._vars[i] != other_vars[i]:
                return False
        return True

    def to_xml(self):
        """Get the XML representation of this object.

        Returns:
            (xml.etree.ElementTree) XML element representing this object
        """
        root = xml_tree.Element("michiganSolution")
        root.append(self._rule.to_xml())
        attributes = xml_tree.SubElement(root, "attributes")
        for key, value in self.get_attributes().items():
            attribute = xml_tree.SubElement(attributes, "attribute")
            attribute.set("attributeID", key)
            attribute.text = str(value)

        return root


    cpdef void set_antecedent_knowledge(self, Knowledge new_knowledge):
        self.get_antecedent().set_knowledge(new_knowledge)