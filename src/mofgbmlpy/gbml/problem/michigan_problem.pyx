from pymoo.core.problem import Problem
import numpy as np
cimport numpy as cnp


class MichiganProblem(Problem):
    """Michigan problem used by an optimization algorithm

    Attributes:
        __training_ds (Dataset): Training dataset
        __rule_builder (RuleBuilderCore): Rule builder for the Michigan solutions
        __objectives (ObjectiveFunction[]): Array of objectives
        __num_constraints (int): Number of constraints (not used yet in the current version)
    """

    def __init__(self,
                 objectives,
                 num_constraints,
                 training_dataset,
                 rule_builder):
        """Constructor

        Args:
            objectives (ObjectiveFunction[]): Array of objectives
            num_constraints (int): Number of constraints (not used yet in the current version)
            training_dataset (Dataset): Training dataset
            rule_builder (RuleBuilderCore): Rule builder for the Michigan solutions
        """

        super().__init__(n_var=1, n_obj=len(objectives))  # 1 var because we consider one solution object
        self.__training_ds = training_dataset
        self.__rule_builder = rule_builder
        self.__objectives = objectives
        self.__num_constraints = num_constraints

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
            self.__objectives[i].run(X[:, 0], i, eval_values[:, i])
        out["F"] = eval_values

    def get_num_objectives(self):
        """Get the number of objectives

        Returns:
            int: Number of objectives
        """
        return len(self.__objectives)
