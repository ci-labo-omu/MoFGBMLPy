from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.population import Population
from pymoo.optimize import minimize

from mofgbmlpy.explainer.gbml.operators.fuzzy_sets_survival import FuzzySetsSurvival
from mofgbmlpy.explainer.gbml.problem.counterfactual_problem import CounterfactualProblem
from pymoo.termination import get_termination
from pymoo.visualization.scatter import Scatter
from pyrecorder.recorder import Recorder
from pyrecorder.writers.video import Video
import os
import numpy as np
from mofgbmlpy.explainer.gbml.fuzzy_sets_eliminate_duplicates import FuzzySetsEliminateDuplicates


class CounterFactualExplainerMetaheuristicsAbstract:
    """Counterfactual explainer using metaheuristics to find CF rules.

    Attributes:
        _problem (CounterfactualProblem): The optimization problem to solve, containing the classifier, the changed rule index, the target class, the test set and the objectives to optimize
        _sampling (Sampling): The sampling method to create new solutions with fuzzy sets for the rules
        _mutation (Mutation): The mutation operator to apply to the solutions, should consider bounds and conditions of membership functions params
        _crossover (Crossover): The crossover operator to apply to the solutions, should consider bounds and conditions of membership functions params
        _survival (Survival): The survival selection method to select the solutions for the next generation, should consider crowding in the search space of fuzzy sets to eliminate duplicates
        _eliminate_duplicates (FuzzySetsEliminateDuplicates): The method to eliminate duplicate solutions based on their fuzzy sets, used after the optimization to filter the non-dominated solutions and keep only one solution for each unique fuzzy sets configuration
        _n_gen (int): The number of generations for the optimization
        _pop_size (int): The population size for the optimization
    """

    def __init__(
        self,
        classifier,
        changed_rule_index,
        target_class,
        test_set,
        sampling,
        mutation,
        crossover,
        use_search_space_crowding=False,
        objectives=["confidence_loss", "change_loss"],
        n_gen=60,
        pop_size=60,
    ):
        """Constructor

        Args:
            classifier (Classifier): The classifier for which to find counterfactual explanations, used to get the knowledge for the mutation operator
            changed_rule_index (int): The index of the rule to change in the counterfactual explanation, used to get the initial rule for the sampling and to apply the changes in the mutation and crossover operators
            target_class (int): The target class for the counterfactual explanation, used to calculate the confidence loss objective
            test_set (DataSet): The test set to evaluate the solutions on, used to calculate the confidence loss objective
            sampling (Sampling): The sampling method to create new solutions with fuzzy sets for the rules
            mutation (Mutation): The mutation operator to apply to the solutions, should consider bounds and conditions of membership functions params
            crossover (Crossover): The crossover operator to apply to the solutions, should consider bounds and conditions of membership functions params
            use_search_space_crowding (bool, optional): Whether to use search space crowding instead of objective space crowding. Defaults to False.
            objectives (list of str, optional): The list of objectives to optimize. Defaults to ["confidence_loss", "change_loss"].
            n_gen (int, optional): The number of generations for the optimization. Defaults to 60.
            pop_size (int, optional): The population size for the optimization. Defaults to 60.
        """
        self._problem = CounterfactualProblem(
            classifier, changed_rule_index, target_class, test_set=test_set, objectives=objectives
        )

        self._sampling = sampling
        self._mutation = mutation
        self._crossover = crossover
        self._survival = FuzzySetsSurvival(use_search_space_crowding=use_search_space_crowding)
        self._eliminate_duplicates = FuzzySetsEliminateDuplicates(self._problem)
        self._n_gen = n_gen
        self._pop_size = pop_size

    def get_target_class(self):
        """Get the target class for the counterfactual explanation.

        Returns:
            ClassLabelBasic: The target class for the counterfactual explanation, used to calculate the confidence loss objective
        """
        return self._problem.get_target_class()

    @staticmethod
    def _save_generations_video_pymoo(history, out_path, file_name_without_extension):
        """Save the generations of a pymoo optimization history as a video.

        Args:
            history (list): List of pymoo optimization history objects.
            out_path (str): Path to save the video to.
            title (str): Title of the video.
        """
        os.makedirs(out_path, exist_ok=True)
        out_file_path = os.path.join(out_path, file_name_without_extension + ".mp4")

        with Recorder(Video(out_file_path)) as rec:
            for entry in history:
                sc = Scatter(title=("Gen %s" % entry.n_gen))

                full_pop = entry.pop.get("F")
                opt_pop = entry.opt.get("F")
                full_pop = np.array([x for x in full_pop if x not in opt_pop])

                if len(full_pop) != 0:
                    sc.add(full_pop, color="blue")
                elif len(opt_pop) == 0:
                    raise ValueError("No data to plot")
                sc.add(opt_pop, color="red")

                sc.do()

                rec.record()

    @staticmethod
    def remove_non_target_class_solutions(solutions, target_class):
        """Remove solutions that do not belong to the target class.

        Args:
            solutions (Population): List of solutions to filter.
            target_class (ClassLabelBasic): The target class to keep.

        Returns:
            list: Filtered list of solutions that belong to the target class.
        """
        filtered_solutions_X = []
        filtered_solutions_F = []

        if solutions is None:
            return None

        for i in range(len(solutions)):
            solution = solutions[i]
            rule = solution.X[0]
            if rule.get_class_label() == target_class and not rule.get_class_label().is_rejected():
                filtered_solutions_X.append(solution.X)
                filtered_solutions_F.append(solution.F)

        filtered_solutions_X = np.array(filtered_solutions_X, dtype=object)
        filtered_solutions_F = np.array(filtered_solutions_F, dtype=float)

        return Population.new(X=filtered_solutions_X, F=filtered_solutions_F)

    def train(self, verbose=True):
        """Train the counterfactual explainer by optimizing the objectives using the specified metaheuristic algorithm.

        Args:
            verbose (bool, optional): Whether to print the optimization progress. Defaults to True.

        Returns:
            Population: The non-dominated solutions found by the optimization, filtered to keep only one solution for each unique fuzzy sets configuration and only solutions that belong to the target class.
        """
        # self._problem.get_fuzzy_rule().plot_antecedent()

        termination = get_termination("n_gen", self._n_gen)

        algorithm = NSGA2(
            pop_size=self._pop_size,
            sampling=self._sampling,
            crossover=self._crossover,
            mutation=self._mutation,  # should consider bounds and conditions of membership functions params
            eliminate_duplicates=False,  # self._eliminate_duplicates,
            save_history=False,  # True,
            survival=self._survival,
        )

        res = minimize(self._problem, algorithm, seed=41, verbose=verbose, termination=termination)

        non_dominated_solutions = res.opt

        if non_dominated_solutions is None:
            return Population.new(X=np.array([], dtype=object), F=np.array([], dtype=float))

        non_dominated_solutions = self._eliminate_duplicates.do(non_dominated_solutions)

        if len(non_dominated_solutions) == 0:
            return non_dominated_solutions

        non_dominated_solutions = self.remove_non_target_class_solutions(
            non_dominated_solutions, self.get_target_class()
        )

        return non_dominated_solutions

    def get_problem(self):
        """Get the optimization problem being solved by the counterfactual explainer.

        Returns:
            CounterfactualProblem: The optimization problem being solved by the counterfactual explainer, containing the classifier, the changed rule index, the target class, the test set and the objectives to optimize
        """
        return self._problem
