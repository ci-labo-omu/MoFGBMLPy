# distutils: language = c++
from libcpp.vector cimport vector as cvector

import xml.etree.cElementTree as xml_tree
import copy
import time

import numpy as np
cimport numpy as cnp
from matplotlib import pyplot as plt

from mofgbmlpy.data.class_label.abstract_class_label cimport AbstractClassLabel
from mofgbmlpy.data.dataset cimport Dataset
from mofgbmlpy.data.pattern cimport Pattern
from mofgbmlpy.fuzzy.classification.abstract_classification import AbstractClassification
from mofgbmlpy.fuzzy.rule.consequent.ruleWeight.rule_weight_multi import RuleWeightMulti
from mofgbmlpy.gbml.solution.abstract_solution cimport AbstractSolution
from mofgbmlpy.fuzzy.knowledge.knowledge import Knowledge
from mofgbmlpy.gbml.solution.abstract_solution cimport AbstractSolution
from mofgbmlpy.gbml.solution.michigan_solution cimport MichiganSolution
from mofgbmlpy.gbml.solution.michigan_solution_builder cimport MichiganSolutionBuilder
from mofgbmlpy.fuzzy.rule.antecedent.antecedent cimport Antecedent

from mofgbmlpy.fuzzy.classification.single_winner_rule_selection import SingleWinnerRuleSelection

from mofgbmlpy.fuzzy.rule.rule_builder_basic import RuleBuilderBasic

from mofgbmlpy.fuzzy.rule.consequent.learning.learning_basic import LearningBasic

from mofgbmlpy.fuzzy.rule.antecedent.factory.heuristic_antecedent_factory import HeuristicAntecedentFactory
from mofgbmlpy.gbml.solution.pittsburgh_scikit_classifier import PittsburghScikitClassifier

cdef class PittsburghSolution(AbstractSolution):
    """Pittsburgh solution (which is a classifier)

    Attributes:
        __classification (AbstractClassification): Classification method used for classification
        __michigan_solution_builder (MichiganSolutionBuilder): Michigan solution builder
        _vars (MichiganSolution[]): Variables: Array of Michigan solutions
    """
    def __init__(self, num_vars, num_objectives, num_constraints, classification, michigan_solution_builder=None, do_init_vars=True):
        """Constructor

        Args:
            num_vars (int): Number of variables (Michigan solutions)
            num_objectives (int): Number of objectives
            num_constraints (int): Number of constraints
            classification (AbstractClassification): Classification method used for classification
            michigan_solution_builder (MichiganSolutionBuilder): Michigan solution builder
            do_init_vars (bool): If true then the Michigan solutions are generated now, otherwise it's delayed and set_vars must be used with learning
        """
        super().__init__(num_objectives, num_constraints)
        self.__michigan_solution_builder = michigan_solution_builder
        self.__classification = classification
        self._error_rate = -1
        self._errored_patterns = None
        if do_init_vars:
            if michigan_solution_builder is None:
                raise TypeError("Michigan solution builder can't be None if do init vars is True")
            self._vars = michigan_solution_builder.create(num_vars)

    cpdef MichiganSolutionBuilder get_michigan_solution_builder(self):
        """Get the michigan solution builder
        
        Returns:
            MichiganSolutionBuilder: Michigan solution builder
        """
        if self.__michigan_solution_builder is None:
            raise TypeError("Michigan solution builder was not initialized but is is accessed")
        return self.__michigan_solution_builder


    cpdef void learning(self, Dataset dataset=None):
        """Update the consequent of the michigan solutions
        
        Args:
            dataset (Dataset): If provided, this dataset is instead of the one stored in the Michigan solutions  
        """
        for var in self._vars:
            var.learning(dataset)

    cpdef double get_average_rule_weight(self):
        """Get the average rule weight
        
        Returns:
            double: Average rule weight
        """
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
                                          copy.deepcopy(self.__classification),
                                          copy.deepcopy(self.__michigan_solution_builder),
                                          do_init_vars=False)

        cdef MichiganSolution[:] vars_copy = np.empty(self.get_num_vars(), dtype=object)
        cdef double[:] objectives_copy = np.empty(self.get_num_objectives(), dtype=np.float64)
        cdef int i
        cdef MichiganSolution var

        for i in range(vars_copy.shape[0]):
            var = copy.deepcopy(self._vars[i])
            vars_copy[i] = var

        for i in range(objectives_copy.shape[0]):
            objectives_copy[i] = self._objectives[i]

        errored_patterns_copy = None
        if self._errored_patterns is not None:
            errored_patterns_copy = np.empty(self._errored_patterns.shape[0], dtype=object)
            for i in range(self._errored_patterns.shape[0]):
                errored_patterns_copy[i] = self._errored_patterns[i]

        new_solution._vars = vars_copy
        new_solution._objectives = objectives_copy
        new_solution._error_rate = self._error_rate
        new_solution._errored_patterns = errored_patterns_copy

        memo[id(self)] = new_solution

        return new_solution

    def __copy__(self):
        """Return a copy of this object. This is a deep copy to avoid issues with Pymoo

        Returns:
            object: Deep copy of this object
        """
        return self.__deepcopy__()

    def __hash__(self):
        """Hash function

        Returns:
            int: Hash value
        """
        cdef int i
        cdef int hash_val = 13

        for i in range(len(self._vars)):
            hash_val += hash_val * 17 + hash(self._vars[i]) * 17
        return hash_val

    cpdef void remove_vars(self, int[:] indices):
        """Remove the variable at the given index

        Args:
            indices (int[]): Indices of the variables removed 
        """
        self._vars = np.delete(self._vars, indices)

    cpdef void clear_vars(self):
        """Clear the variables"""
        self._vars = np.empty(0, dtype=object)

    cpdef MichiganSolution[:] get_vars(self):
        """Get the variables
        
        Returns:
            MichiganSolution[]: variables
        """
        return self._vars

    cpdef MichiganSolution get_var(self, int index):
        """Get the variable at the give index
        
        Args:
            index (int): Index of the variable fetched

        Returns:
            MichiganSolution: Variable fetched
        """
        return self._vars[index]

    cpdef void set_var(self, int index, MichiganSolution value):
        """Set the variable at the given index
        
        Args:
            index (int): Index of the variable changed
            value (MichiganSolution): New value 
        """
        self._vars[index] = value

    cpdef void set_vars(self, MichiganSolution[:] new_vars):
        """Set the variables
        
        Args:
            new_vars (MichiganSolution[]): New variables
        """
        self._vars = new_vars

    cpdef int get_num_vars(self):
        """Get the number of variables
        
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
        for i in range(len(self._vars)):
            sol_xml = self._vars[i].to_xml()
            sol_xml.set("id", str(i))
            root.append(sol_xml)

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

    def to_csv(self):
        csv_str = "PittsburghSolution_Basic,"
        for i in range(self.get_num_objectives()):
            csv_str += f"Objectives[{i}]=,{self.get_objective(i):.4f}..,"
        csv_str += "\n"
        for var in self._vars:
            csv_str += f"\t->{var.to_csv()}\n"
        attributes = self.get_attributes()
        csv_str += ",attributes={,"
        for key, value in attributes.items():
            csv_str += f"{key},{value},"
        csv_str += "}"
        return csv_str

    @staticmethod
    def from_xml(xml_element, random_gen, knowledge, num_objectives, num_constraints, rule_builder, classification, michigan_solution_builder):
        imported_rules = xml_element.findall("michiganSolution")
        cdef MichiganSolution[:] new_vars = np.empty(len(imported_rules), dtype=object)
        cdef MichiganSolution var

        for i in range(len(imported_rules)):
            var = MichiganSolution.from_xml(imported_rules[i], random_gen, 2, 0, rule_builder, knowledge)
            # idx = int(imported_rules[i].get("id"))
            new_vars[i] = var

        num_vars = len(new_vars)
        new_sol = PittsburghSolution(num_vars, num_objectives, num_constraints, classification, michigan_solution_builder)
        new_sol.set_vars(new_vars)

        attributes_xml = xml_element.find("attributes")
        if attributes_xml is not None:
            for attributes_xml in attributes_xml.findall("attribute"):
                attr_id = attributes_xml.get("attributeID")
                attr_value = attributes_xml.text
                new_sol.set_attribute(attr_id, attr_value)
        return new_sol

    cpdef bint are_rules_valid(self):
        """Check if the rules are valid (at least one rule inside this solution and no rejected rule)
        
        Returns:
            bool: True if they are valid and false otherwise
        """
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
        """Get the Michigan solution used to classify a pattern using the classification method of this solution. Please refer to the classify method of the classification methods for more details. Can only be accessed from Cython code.
        
        Args:
            pattern (Pattern): Pattern used to select a Michigan solution (a winner) 

        Returns:
            MichiganSolution: The winner if there is any and None otherwise.
        """
        return self.__classification.classify(self._vars, pattern)

    cpdef MichiganSolution classify_py(self, Pattern pattern):
        """Get the Michigan solution used to classify a pattern using the classification method of this solution. Please refer to the classify method of the classification methods for more details.
 
        Args:
            pattern (Pattern): Pattern used to select a Michigan solution (a winner) 

        Returns:
            MichiganSolution: The winner if there is any and None otherwise.
        """
        return self.classify(pattern)

    cpdef AbstractClassLabel predict(self, Pattern pattern):
        """Predict the class of a pattern
        
        Args:
            pattern (Pattern): Pattern whose class is predicted 

        Returns:
            AbstractClassLabel: Class predicted
        """
        cdef MichiganSolution winner = self.classify(pattern)
        if winner is None:
            return None
        else:
            return winner.get_class_label()

    cpdef get_total_rule_length(self):
        """Get the total rule length (sum of the rule lengths of all Michigan solutions in this classifier)
        
        Returns:
            int: Total rule length
        """
        length = 0
        if self._vars is not None:
           for item in self._vars:
               length += item.get_length()
        return length

    cpdef void update_winners_and_errors(self, Dataset dataset):
        """Update the number of wins and errors of the Michigan solutions in this classifier.
        
        Args:
            dataset (Dataset): Dataset used to update the number of wins and errors of the Michigan solutions
        """
        if self._vars is None or dataset is None:
           raise TypeError("Michigan solutions list and dataset can't be None")

        cdef int num_errors = 0
        cdef int dataset_size = dataset.get_size()
        cdef int i
        cdef MichiganSolution winner_solution
        cdef Pattern[:] patterns = dataset.get_patterns()
        cdef Pattern p
        cdef cvector[int] errored_patterns_indices
        cdef object[:] errored_patterns

        for sol in self._vars:
           sol.reset_num_wins()
           sol.reset_fitness()

        for i in range(dataset.get_size()):
           p = patterns[i]
           winner_solution = self.classify(p)
           if winner_solution is None:
               # If output is rejected then continue next pattern.
               num_errors += 1
               errored_patterns_indices.push_back(i)
               continue

           winner_solution.inc_num_wins()

           if p.get_target_class() != winner_solution.get_class_label():
               num_errors += 1
               errored_patterns_indices.push_back(i)
           else:
               winner_solution.inc_fitness()

        self._errored_patterns = np.empty(errored_patterns_indices.size(), dtype=object)
        for i in range(errored_patterns_indices.size()):
           self._errored_patterns[i] = patterns[errored_patterns_indices[i]]

        for sol in self._vars:
            sol.set_scores_update_status(True)

        self._error_rate = num_errors / dataset_size

    cpdef double calc_error_rate(self, Dataset dataset):
        """Calculate the error rate of this classifier using the given dataset

        Args:
            dataset (Dataset): Dataset used to calculate the error rate

        Returns:
            double: Error rate
        """
        if self._vars is None or dataset is None:
           raise TypeError("Michigan solutions list and dataset can't be None")

        cdef int num_errors = 0
        cdef int dataset_size = dataset.get_size()
        cdef int i
        cdef MichiganSolution winner_solution
        cdef Pattern[:] patterns = dataset.get_patterns()
        cdef Pattern p

        # for sol in self._vars:
        #    sol.reset_num_wins()
        #    sol.reset_fitness()

        for i in range(dataset.get_size()):
           p = patterns[i]
           winner_solution = self.classify(p)
           if winner_solution is None or p.get_target_class() != winner_solution.get_class_label():
               num_errors += 1

        return num_errors / dataset_size

    cpdef double get_error_rate(self):
        """Get the error rate of the last update.
    
        Returns:
            double: Error rate
        """
        if self._error_rate == -1:
            raise Exception("Error rate was not initialized. Please call update_winners_and_errors first")
        return self._error_rate


    cpdef object[:] get_errored_patterns(self):
        """Get the patterns that can't be classified by this classifier in the last update
 
        Returns:
            double: Errored patterns
        """

        return self._errored_patterns


    cpdef AbstractClassification get_classification(self):
        """Get the classification method
        
        Returns:
            AbstractClassification: Classification method object
        """
        return self.__classification

    def create_scikit_classifier(self):
        """Create a Scikit-learn classifier (an Estimator) corresponding to this classifier

        Returns:
            PittsburghScikitClassifier: New Scikit-learn classifier
        """
        return PittsburghScikitClassifier(self)

    def plot_rules(self, var_names=None, rules_names=None, dims=None):
        """Plot the rules of this classifier

        Args:
            var_names (list): Names of the variables (Michigan solutions) to use in the plot
            rules_names (list): Names of the rules to use in the plot
            dims (list): Dimensions to plot. If None then all dimensions are plotted
        """
        cdef int num_vars = self.get_num_vars()
        cdef int i
        cdef MichiganSolution var
        cdef Antecedent antecedent

        antecedent = self.get_var(0).get_antecedent()

        if dims is None:
            dims = list(range(antecedent.get_array_size()))
        else:
            prev_dim = None
            for dim in dims:
                if dim < 0 or dim >= antecedent.get_array_size():
                    raise ValueError(f"Dimension {dim} is out of range")
                if prev_dim is not None and dim <= prev_dim:
                    raise ValueError("Dimensions must be in ascending order and unique")
                prev_dim = dim

        num_dims = len(dims)

        if rules_names is None:
            rules_names = []
            for i in range(num_vars):
                rules_names.append(f"Rule {i}")

        fig, axes = plt.subplots(num_vars, num_dims, figsize=(4*num_dims, 3*num_vars))
        fig.subplots_adjust(hspace=1.0, wspace=0.35)

        for i in range(num_vars):
            var = self.get_var(i)
            antecedent = var.get_antecedent()
            axes[i, :] = antecedent.get_plot(axes[i, :], var_names, dims, line_width=5)

            row_bbox = axes[i, 0].get_position()
            y_center = 0.5 * (row_bbox.y0 + row_bbox.y1)

            y = axes[i, 0].get_position().y1 + 0.01*num_dims

            fig.text(
                0.5, y,
                rules_names[i],
                ha="center",
                va="bottom",
                fontsize=20,
                fontweight="bold"
            )

        plt.show()