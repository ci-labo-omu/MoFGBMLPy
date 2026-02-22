"""Pymoo problem class for the task offloading problem."""

import copy

import numpy as np
from pymoo.core.problem import Problem
from mofgbmlpy.fuzzy.knowledge.knowledge import Knowledge
from mofgbmlpy.fuzzy.fuzzy_term.fuzzy_variable import FuzzyVariable
from mofgbmlpy.fuzzy.fuzzy_term.fuzzy_set.dont_care_fuzzy_set import DontCareFuzzySet

from mofgbmlpy.explainer.util import append_rule_classifier


class CounterfactualProblem(Problem):
    """Counterfactual problem for finding counterfactual explanations for a given factual rule in a classifier.

    Attributes:
        _classifier_copy_mutable (PittsburghSolution): A mutable copy of the classifier to modify
        _classifier_copy (PittsburghSolution): A copy of the classifier to use for computing the error rate
        _changed_rule_index (int): The index of the rule in the classifier that is being changed
        _factual_michigan_solution (MichiganSolution): The factual Michigan solution
        _initial_class (ClassLabelBasic): The class label of the factual Michigan solution.
        _target_class (ClassLabelBasic): The target class label for the counterfactual explanation.
        _area_computation_num_samples (int): The number of samples for computing the area under the membership functions
        _learner (AbstractLearning): The learner of the factual Michigan solution, used for the confidence loss.
        _train_set (Dataset): The training dataset of the factual Michigan solution, used for the confidence loss.
        _test_set (Dataset): The test dataset to evaluate the error rate of the counterfactual solutions.
        _initial_mfs_y (np.array): The membership function values of the factual rule, used for the change loss.
        _objectives_map (dict): A dictionary mapping the names of the objectives to their corresponding functions.
    """

    def __init__(
        self, classifier, changed_rule_index, target_class, test_set, objectives=["confidence_loss", "change_loss"]
    ):
        """Constructor

        Args:
            classifier (PittsburghSolution): The classifier to which the factual rule belongs.
            changed_rule_index (int): The index of the rule in the classifier that is being changed.
            target_class (ClassLabelBasic): The target class label for the counterfactual explanation.
            test_set (Dataset): The test dataset to evaluate the error rate of the counterfactual solutions.
            objectives (list): A list of the names of the objectives to optimize. Possible values are "confidence_loss", "change_loss", "num_changed_features", "train_error_rate" and "rule_length".
        """
        self._classifier_copy_mutable = copy.deepcopy(classifier)
        self._classifier_copy = copy.deepcopy(classifier)
        self._changed_rule_index = changed_rule_index
        self._factual_michigan_solution = classifier.get_var(changed_rule_index)
        self._initial_class = self._factual_michigan_solution.get_class_label().get_class_label_value()
        self._target_class = target_class
        self._area_computation_num_samples = 50  # The higher it is, the more precise it gets, but it's also slower
        self._learner = self._factual_michigan_solution.get_rule_builder().get_consequent_factory()
        self._train_set = self._learner.get_training_set()
        self._test_set = test_set

        self._initial_mfs_y = self.compute_membership_values(self._factual_michigan_solution, 0, 1)

        self._objectives_map = {
            "confidence_loss": self.conf_loss,
            "change_loss": self.change_loss,
            "num_changed_features": self.num_changed_features_loss,
            "train_error_rate": self.error_rate,
            "rule_length": lambda rule: rule.get_length(),
        }

        self._objectives_map = {k: v for k, v in self._objectives_map.items() if k in objectives}

        if len(self._objectives_map) == 0:
            raise ValueError("At least one objective must be selected")

        super().__init__(n_var=1, n_obj=len(self._objectives_map), n_eq_constr=1)

    def get_factual_rule(self):
        """Get the factual Michigan solution.

        Returns:
            MichiganSolution: The factual Michigan solution.
        """
        return self._factual_michigan_solution

    def get_initial_num_rules(self):
        """Get the initial number of rules in the classifier.

        Returns:
            int: The initial number of rules in the classifier.
        """
        return self._classifier_copy.get_num_vars()

    def get_target_class(self):
        """Get the target class label for the counterfactual explanation.

        Returns:
            ClassLabelBasic: The target class label for the counterfactual explanation.
        """
        return self._target_class

    @staticmethod
    def get_fuzzy_sets_from_rule(rule):
        """Get the fuzzy sets of a rule as an array.

        Args:
            rule (Rule): The rule to get the fuzzy sets from.

        Returns:
            np.array: An array of the fuzzy sets of the rule.
        """
        fs_list = [rule.get_fuzzy_set_object(dim) for dim in range(rule.get_antecedent_array_size())]
        return np.array(fs_list, dtype=object)

    def get_initial_fuzzy_sets(self):
        """Get the fuzzy sets of the factual rule as an array.

        Returns:
            np.array: An array of the fuzzy sets of the factual rule.
        """
        return CounterfactualProblem.get_fuzzy_sets_from_rule(self._factual_michigan_solution.get_rule())

    def compute_membership_values(self, rule, min_val=0, max_val=1):
        """Compute the membership function values of a rule for a given range of attribute values.

        Args:
            rule (Rule): The rule to compute the membership function values for.
            min_val (float): The minimum value of the attribute range. Default is 0.
            max_val (float): The maximum value of the attribute range. Default is 1.

        Returns:
            np.array: An array of the membership function values of the rule for the given range of attribute values.
        """
        fuzzy_sets = CounterfactualProblem.get_fuzzy_sets_from_rule(rule.get_rule())
        mfs = [fs.get_function() for fs in fuzzy_sets]

        x_samples = np.linspace(min_val, max_val, self._area_computation_num_samples)
        mf_values = np.zeros((len(fuzzy_sets), len(x_samples)), dtype=object)
        for i, mf in enumerate(mfs):
            mf_values[i] = [mf.get_value_py(x) for x in x_samples]

        return mf_values

    def compute_iou_with_factual(self, rule):
        """Compute the Intersection over Union (IoU) between the membership functions of a rule and the factual rule.

        Args:
            rule (Rule): The rule to compute the IoU with the factual rule.

        Returns:
            np.array: An array of the IoU values for each fuzzy set of the rule.
        """
        step = 1 / self._area_computation_num_samples

        mf_1_y = self._initial_mfs_y
        mf_2_y = self.compute_membership_values(rule)

        union_value = np.zeros(mf_1_y.shape[0])
        intersection_value = np.zeros(mf_1_y.shape[0])

        for fs_i in range(mf_1_y.shape[0]):
            intersection_value[fs_i] = 0
            union_value[fs_i] = 0

            for i in range(mf_1_y.shape[1]):
                y_val = max(mf_1_y[fs_i][i], mf_2_y[fs_i][i])
                union_value[fs_i] += step * y_val

                y_val = min(mf_1_y[fs_i][i], mf_2_y[fs_i][i])
                intersection_value[fs_i] += step * y_val

            if union_value[fs_i] == 0:
                union_value[fs_i] = 1
                intersection_value[fs_i] = 1

        return intersection_value / union_value

    @staticmethod
    def build_knowledge(fuzzy_sets):
        """Build a Knowledge object from an array of fuzzy sets.

        Args:
            fuzzy_sets (np.array): An array of fuzzy sets to build the knowledge from.

        Returns:
            Knowledge: A Knowledge object built from the given fuzzy sets.
        """
        antecedent_indices = np.ones(len(fuzzy_sets), dtype=int)
        fuzzy_vars = np.empty(len(fuzzy_sets), dtype=object)

        for i in range(len(fuzzy_sets)):
            if fuzzy_sets[i] is None or len(fuzzy_sets[i].get_function().get_params()) == 0:
                # DC
                antecedent_indices[i] = 0
                fuzzy_vars[i] = FuzzyVariable(fuzzy_sets=np.array([DontCareFuzzySet(0)]), name=f"x{i}")
            else:
                fuzzy_vars[i] = FuzzyVariable(fuzzy_sets=np.array([DontCareFuzzySet(0), fuzzy_sets[i]]), name=f"x{i}")
        knowledge = Knowledge(fuzzy_vars)

        return knowledge

    def conf_loss(self, current_rule):
        """Compute the confidence loss of a rule based on the confidence of the target class.

        Args:
            current_rule (Rule): The rule to compute the confidence loss for.

        Returns:
            float: The confidence loss of the rule.
        """
        # Confidence loss

        antecedent = current_rule.get_antecedent()
        confidences = self._learner.calc_confidence_py(antecedent, self._train_set)

        confidence_target_class = confidences[self._target_class.get_class_label_value()]

        confidence_loss = 1 - confidence_target_class

        return confidence_loss

    def change_loss(self, current_rule):
        """Compute the change loss of a rule based on the IoU between it and the factual rule.

        Args:
            current_rule (Rule): The rule to compute the change loss for.

        Returns:
            float: The change loss of the rule.
        """
        # Change loss
        change_loss = 0

        iou = self.compute_iou_with_factual(current_rule)

        if iou is not None:
            change_loss = 1 - np.mean(iou)

        return change_loss

    def num_changed_features_loss(self, current_michigan_solution):
        """Compute the number of changed features loss of a Michigan solution based on the number of different fuzzy sets between it and the factual rule.

        Args:
            current_michigan_solution (MichiganSolution): The Michigan solution to compute the number of changed features loss for.

        Returns:
            int: The number of changed features loss of the Michigan solution.
        """
        num_changed_features = 0
        current_rule = current_michigan_solution.get_rule()
        factual_rule = self._factual_michigan_solution.get_rule()

        for i in range(current_rule.get_antecedent_array_size()):
            fs1 = factual_rule.get_fuzzy_set_object(i)
            fs2 = current_rule.get_fuzzy_set_object(i)

            if CounterfactualProblem.are_fuzzy_set_different(fs1, fs2):
                num_changed_features += 1

        return num_changed_features

    def _create_new_classifier(self, sol, replace=True):
        """Create a new classifier by replacing the rule with the given solution or by appending it to the classifier.

        Args:
            sol (MichiganSolution): The Michigan solution to use for creating the new classifier.
            replace (bool): Whether to replace the changed rule with the solution or to append it to the classifier.

        Returns:
            PittsburghSolution: A new classifier with the added solution.
        """
        new_classifier = self._classifier_copy_mutable if replace else copy.deepcopy(self._classifier_copy)

        if replace:
            new_classifier.set_var(self._changed_rule_index, sol)
        else:
            new_classifier = append_rule_classifier(new_classifier, sol, train_set=self._train_set, deepcopy=False)
        return new_classifier

    def error_rate(self, current_rule=None, use_test_set=False, replace=True, initial_classifier=False):
        """Compute the error rate of a rule by creating a new classifier with the rule and evaluating it on the test set

        Args:
            current_rule (Rule): The rule to compute the error rate for. If None, the error rate of the initial classifier is computed.
            use_test_set (bool): Whether to use the test set or the training set for computing the error rate.
            replace (bool): Whether to replace the changed rule with the given solution or to append it to the classifier.
            initial_classifier (bool): Whether to compute the error rate of the initial classifier before any changes.
        """
        dataset = self._test_set if use_test_set else self._train_set

        if initial_classifier:
            return self._classifier_copy.calc_error_rate(self._train_set)

        if current_rule.get_rule().is_rejected_class_label():
            return 1.0

        new_classifier = self._create_new_classifier(current_rule, replace=replace)

        return new_classifier.calc_error_rate(dataset)

    def is_output_class_target(self, current_rule):
        """Check if the output class label of a rule is the target class label.

        Args:
            current_rule (Rule): The rule to check.

        Returns:
            bool: True if the output class label of the rule is the target class label, False otherwise.
        """
        return current_rule.get_class_label() == self._target_class

    def get_objectives(self, current_rule):
        """Get the values of the objectives for a given rule.

        Args:
            current_rule (Rule): The rule to compute the objectives for.

        Returns:
            np.array: An array of the values of the objectives for the given rule.
        """
        objectives = np.empty(self.n_obj)
        for i, func in enumerate(self._objectives_map.values()):
            objectives[i] = func(current_rule)

        return objectives

    def _evaluate(self, X, out, *args, **kwargs):
        """Evaluate the objectives and constraints for a given set of solutions.

        Args:
            X (np.array): An array of solutions to evaluate.
            out (dict): A dictionary to store the evaluated objectives and constraints.
        """
        out["F"] = np.empty((len(X), self.n_obj))
        out["H"] = np.empty((len(X),))

        for i, ind in enumerate(X):
            ind[0].learning()
            rule = ind[0]
            for j in range(self.n_obj):
                out["F"][i] = self.get_objectives(rule)
                rule.set_objective(j, out["F"][i][j])
            out["H"][i] = (
                0 if self.is_output_class_target(rule) else 1
            )  # constraint, note that rejected class labels are also removed here

    def get_objective_names(self):
        """Get the names of the objectives.

        Returns:
            list: A list of the names of the objectives.
        """
        return list(self._objectives_map.keys())

    def num_wins_and_successes(self, sol=None, replace=False, initial_classifier=False):
        """Get the number of wins and fitness values of the rules in the classifier.

        Args:
            sol (MichiganSolution): The Michigan solution to evaluate. If None, the current classifier is evaluated.
            replace (bool): Whether to replace the changed rule with the solution or to append it to the classifier.
            initial_classifier (bool): Whether to evaluate the initial classifier before any changes. Default is False.

        Returns:
            np.array: An array of the number of wins of the rules in the classifier.
            np.array: An array of the fitness values of the rules in the classifier.
        """
        # success is defined as fitness in the code of MoFGBML

        if initial_classifier:
            new_classifier = self._classifier_copy
        else:
            new_classifier = self._create_new_classifier(sol, replace=replace)
            new_classifier.update_winners_and_errors(self._train_set)

        cl_vars = new_classifier.get_vars()
        num_wins = np.empty(len(cl_vars), dtype=int)
        fitness_vals = np.empty(len(cl_vars), dtype=int)

        for i, rule in enumerate(cl_vars):
            num_wins[i] = rule.get_num_wins()
            fitness_vals[i] = rule.get_fitness()

        return num_wins, fitness_vals

    @staticmethod
    def are_fuzzy_set_different(fs1, fs2, threshold=1e-8):
        """Check if two fuzzy sets are different based on their parameters.

        Args:
            fs1 (FuzzySet): The first fuzzy set to compare.
            fs2 (FuzzySet): The second fuzzy set to compare.
            threshold (float): The threshold for considering two parameters as different. Default is 1e-8.

        Returns:
            bool: True if the fuzzy sets are different, False otherwise.
        """
        params_1 = fs1.get_function().get_params()
        params_2 = fs2.get_function().get_params()

        if params_1 is None:
            params_1 = []
        if params_2 is None:
            params_2 = []

        if len(params_1) != len(params_2):
            return True

        for i in range(len(params_1)):
            if abs(params_1[i] - params_2[i]) > threshold:
                return True

        return False

    def get_changed_rule_index(self):
        """Get the index of the changed rule in the classifier.

        Returns:
            int: The index of the changed rule in the classifier.
        """
        return self._changed_rule_index

    def get_train_set(self):
        """Get the training set of the factual rule (used for computing the confidence).

        Returns:
            Dataset: The training set of the factual rule.
        """
        return self._train_set

    def get_learner(self):
        """Get the learner of the factual rule (used for computing the confidence).

        Returns:
            AbstractLearning: The learner of the factual rule.
        """
        return self._learner
