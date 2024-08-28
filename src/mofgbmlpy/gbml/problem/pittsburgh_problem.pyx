import sys
import copy

from pymoo.core.population import Population
from pymoo.core.problem import Problem
import numpy as np
cimport numpy as cnp

from mofgbmlpy.exception.empty_pittsburgh_solution import EmptyPittsburghSolution
from mofgbmlpy.gbml.solution.pittsburgh_solution import PittsburghSolution
import time
import cython

class PittsburghProblem(Problem):
    """Pittsburgh problem used by an optimization algorithm

    Attributes:
        __num_vars (int): Number of variables in the Michigan solutions
        __objectives (ObjectiveFunction[]): Array of objectives
        __num_constraints (int): Number of constraints (not used yet in the current version)
        __training_ds (Dataset): Training dataset
        __michigan_solution_builder (MichiganSolutionBuilder): Builder for Michigan solutions
        __classification (Classification): Classification method
    """
    def __init__(self,
                 num_vars,
                 objectives,
                 num_constraints,
                 training_dataset,
                 michigan_solution_builder,
                 classification):

        """Constructor

        Args:
            num_vars (int): Number of variables in the Michigan solutions
            objectives (ObjectiveFunction[]): Array of objectives
            num_constraints (int): Number of constraints (not used yet in the current version)
            training_dataset (Dataset): Training dataset
            michigan_solution_builder (MichiganSolutionBuilder): Builder for Michigan solutions
            classification (Classification): Classification method
        """

        super().__init__(n_var=1, n_obj=len(objectives))  # 1 var because we consider one solution object
        self.__training_ds = training_dataset
        self.__num_vars = num_vars
        self.__michigan_solution_builder = michigan_solution_builder
        self.__classification = classification
        self.__objectives = objectives
        self.__num_constraints = num_constraints
        if len(objectives) == 0:
            raise ValueError("At least one objective is needed")

    def create_solution(self):
        """Create a Pittsburgh solution

        Returns:
            PittsburghSolution: New Pittsburgh solution
        """
        pittsburgh_solution = PittsburghSolution(self.__num_vars,
                                                 self.get_num_objectives(),
                                                 self.__num_constraints,
                                                 self.__classification,
                                                 copy.deepcopy(self.__michigan_solution_builder))

        return pittsburgh_solution

    def get_num_vars(self):
        """Get the number of variables (number of antecedent indices in the Michigan solutions)

        Returns:
            int: Number of variables
        """
        return self.__num_vars

    def get_num_objectives(self):
        """Get the number of objectives

        Returns:
            int: Number of objectives
        """
        return len(self.__objectives)

    def get_num_constraints(self):
        """Get the number of constraints

        Returns:
            int: Number of constraints
        """
        return self.__num_constraints

    def get_training_set(self):
        """Get the training set

        Returns:
            Dataset: Training set
        """
        return self.__training_ds

    def get_rule_builder(self):
        """Get the rule builder

        Returns:
            RuleBuilderCore: Rule builder
        """
        self.__michigan_solution_builder.get_rule_builder()

    def _evaluate(self, X, out, *args, **kwargs):
        """Evaluate the solutions in the population

        Args:
            X (Population): Population evaluated
            out (double[,]): Objective function values for each solution
            *args (tuple): Other arguments for Pymoo
            **kwargs (dict): Other arguments for Pymoo
        """
        cdef cnp.ndarray[double, ndim=2] eval_values = np.empty((len(X), self.get_num_objectives()), dtype=np.float64)
        cdef int i

        for i in range(len(self.__objectives)):
            self.__objectives[i].run(X[:, 0], i, eval_values[:,i])
        out["F"] = eval_values


    def remove_no_winner_michigan_solution(self, solutions):
        """Remove the Michigan solutions with no wins in each one of the Pittsburgh solutions in the given list

        Args:
            solutions (PittsburghSolution[]): List of Pittsburgh solutions

        Returns:
            PittsburghSolution[]: List of Pittsburgh solutions after removal
        """
        for i in range(len(solutions)):
            sol = solutions[i][0]

            # Update eval values
            sol.get_error_rate(self.__training_ds)

            k = 0
            for j in range(sol.get_num_vars()):
                if sol.get_var(k).get_num_wins() < 1:
                    sol.remove_var(k)
                else:
                    k += 1

            if sol.get_num_vars() == 0:
                raise EmptyPittsburghSolution()

        return solutions