import xml.etree.cElementTree as xml_tree
import copy

import numpy as np

from mofgbmlpy.data.dataset cimport Dataset
from mofgbmlpy.data.pattern cimport Pattern
from mofgbmlpy.exception.exceeded_max_trials_number import ExceededMaxTrialsNumber
from mofgbmlpy.fuzzy.knowledge.knowledge cimport Knowledge
from mofgbmlpy.fuzzy.rule.rule_builder_core cimport RuleBuilderCore
from mofgbmlpy.fuzzy.rule.abstract_rule cimport AbstractRule
from mofgbmlpy.fuzzy.rule.antecedent.antecedent cimport Antecedent
from mofgbmlpy.fuzzy.rule.consequent.ruleWeight.abstract_rule_weight cimport AbstractRuleWeight
from mofgbmlpy.gbml.solution.abstract_solution cimport AbstractSolution
cimport numpy as cnp


cdef class MichiganSolution(AbstractSolution):
    """Michigan solution

    Attributes:
        _rule (AbstractRule): Rule
        _rule_builder (RuleBuilderCore): Rule builder
        _vars (int[]): Variables (antecedent indices)
        __num_wins (int): Number of times this solution was selected for classification for one round (on a whole dataset)
        __fitness (int): Fitness value (computed in the error rate calculation). of times this solution was selected for classification and that the classification was right, for one round (on a whole dataset)
        _random_gen (numpy.random.Generator): Random generator
    """
    def __init__(self, random_gen, num_objectives, num_constraints, rule_builder, pattern=None, do_init_vars=True):
        """Constructor

        Args:
            random_gen (numpy.random.Generator): Random generator
            num_objectives (int): Number of objectives
            num_constraints (int): Number of constraints
            rule_builder (RuleBuilderCore): Rule builder
            pattern (Pattern): Pattern used to generate rules
            do_init_vars (bool): If true then the rule is generated now, otherwise it's delayed and set_vars must be used with learning
        """
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
                        raise ExceededMaxTrialsNumber("Exceeded maximum number of trials to generate rule")
            else:
                training_data_set = self.get_rule_builder().get_training_dataset()
                size_training_data_set = training_data_set.get_size()
                self.create_rule(pattern=pattern)
                is_rejected = self._rule.is_rejected_class_label()

                while is_rejected:
                    cnt += 1
                    self.create_rule(training_data_set.get_pattern(random_gen.integers(0, size_training_data_set)))
                    is_rejected = self._rule.is_rejected_class_label()
                    if cnt > 1000:
                        raise ExceededMaxTrialsNumber("Exceeded maximum number of trials to generate rule")

    cdef void create_rule(self, Pattern pattern=None):
        """Create the rule of this solution
        
        Args:
            pattern (Pattern): If provided, use this pattern to create the rule 
        """
        cdef int[:] antecedent_indices
        antecedent_indices = self._rule_builder.create_antecedent_indices(num_rules=1, pattern=pattern)[0]
        self.set_vars(antecedent_indices)
        self.learning()

    cpdef void learning(self, Dataset dataset=None):
        """Update the antecedent object using the vars array (antecedent indices) and re-learn the consequent
        
        Args:
            dataset (Dataset): Training dataset. If not provided, use the one stored in the class 
        """
        cdef Antecedent antecedent_object
        if self._vars is None:
            raise TypeError("Vars is not defined")

        if self._rule is None or self._rule.get_antecedent() is None:
            antecedent_object = self._rule_builder.create_antecedent_from_indices(self._vars)
            self._rule = self._rule_builder.create(antecedent_object)
        else:
            antecedent_object = self._rule.get_antecedent()
            antecedent_object.set_antecedent_indices(self._vars)
            self._rule.set_consequent(self._rule_builder.create_consequent(antecedent_object, dataset))

    cpdef double get_fitness_value(self, double[:] in_vector):
        """Get the fitness value for the given attribute vector
        
        Args:
            in_vector (double[]): Vector for which the fitness value is returned

        Returns:
            double: Fitness value
        """
        return self._rule.get_fitness_value(in_vector)

    cpdef int get_length(self):
        """Get the rule length (length of the antecedent (Number of not DC fuzzy sets inside))
        
        Returns:
            int: Rule length
        """
        return self._rule.get_length()

    cpdef get_class_label(self):
        """Get the class label object of the rule
        
        Returns:
            AbstractClassLabel: Label
        """
        return self._rule.get_class_label()

    cdef AbstractRuleWeight get_rule_weight(self):
        """Get the rule weight object. Can only be used from Cython
        
        Returns:
            AbstractRuleWeight: Rule weight object
        """
        return self._rule.get_rule_weight()

    cpdef AbstractRuleWeight get_rule_weight_py(self):
        """Get the rule weight object

        Returns:
            AbstractRuleWeight: Rule weight object
        """
        return self.get_rule_weight()

    cpdef AbstractRule get_rule(self):
        """Get the rule of this solution
        
        Returns:
            AbstractRule: Rule
        """
        return self._rule

    cpdef RuleBuilderCore get_rule_builder(self):
        """Get the rule builder of this solution

        Returns:
            RuleBuilderCore: Rule
        """
        return self._rule_builder

    cpdef AbstractConsequent get_consequent(self):
        """Get the consequent of this solution

        Returns:
            AbstractConsequent: Consequent
        """
        return self._rule.get_consequent()

    cpdef Antecedent get_antecedent(self):
        """Get the antecedent of this solution

        Returns:
            Antecedent: Antecedent
        """
        return self._rule.get_antecedent()

    cdef double[:] get_membership_values(self, double[:] attribute_vector):
        """Get the membership values for the given vector

        Returns:
            double[]: Membership values
        """
        return self._rule.get_membership_values(attribute_vector)

    cdef double get_compatible_grade_value(self, double[:] attribute_vector):
        """Get the compatible grade value for the given vector
        
        Args:
            attribute_vector (double[]): Attribute vector 

        Returns:
            double: Compatible grade value
        """
        return self._rule.get_compatible_grade_value(attribute_vector)

    cpdef void reset_num_wins(self):
        """Reset to 0 the number of wins"""
        self.__num_wins = 0

    cpdef void reset_fitness(self):
        """Reset to 0 the fitness value"""
        self.__fitness = 0

    cpdef void inc_num_wins(self):
        """Add one to the number of wins"""
        self.__num_wins += 1

    cpdef void inc_fitness(self):
        """Add one to the fitness value"""
        self.__fitness += 1

    cpdef int get_num_wins(self):
        """get the number of wins
        
        Returns:
            int: Number of wins
        """
        return self.__num_wins

    cpdef int get_fitness(self):
        """Get the fitness value
        
        Returns:
            int: Fitness value
        """
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
        """Hash function

        Returns:
            int: Hash value
        """
        cdef int i
        cdef int hash_val = 13

        for i in range(len(self._vars)):
            hash_val += hash_val * 17 + self._vars[i]
        return hash_val

    cdef void clear_vars(self):
        """Clear the variables """
        self._vars = np.empty(0, dtype=int)

    cpdef int[:] get_vars(self):
        """Get the array of variables
        
        Returns:
            int[]: The solution variables
        """
        return self._vars

    cpdef int get_var(self, int index):
        """Get the variable at the given index
        
        Args:
            index (int): Index of the variable fetched 

        Returns:
            int: Variable value
        """
        return self._vars[index]

    cpdef void set_var(self, int index, int value):
        """Set the variable at the given index
        
        Args:
            index (int): Index of the variable changed
            value (int): New variable value 
        """
        self._vars[index] = value

    cpdef void set_vars(self, int[:] new_vars):
        """Set the variables (antecedent indices)
        
        Args:
            new_vars (int[]): New vars 
        """
        self._vars = new_vars

    cpdef int get_num_vars(self):
        """Get the number of variables (number of antecedent indices, i.e. the number of dimensions)
        
        Returns:
            int: Number of variables
        """
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
        """Set the antecedent knowledge base
        
        Args:
            new_knowledge (Knowledge): New knowledge base
        """
        self.get_antecedent().set_knowledge(new_knowledge)