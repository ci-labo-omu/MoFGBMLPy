import copy
from pymoo.core.mutation import Mutation

from mofgbmlpy.data.dataset cimport Dataset
from mofgbmlpy.fuzzy.rule.consequent.learning.learning_basic import LearningBasic
from mofgbmlpy.gbml.solution.michigan_solution cimport MichiganSolution
from mofgbmlpy.gbml.solution.pittsburgh_solution cimport PittsburghSolution


class PittsburghMutation(Mutation):
    """Pittsburgh mutation operator

    Attributes:
        __knowledge (Knowledge): Knowledge base
        _random_gen (numpy.random.Generator): Random generator
        __mutation_rate (float): Mutation rate
    """
    def __init__(self, knowledge, random_gen, mutation_rate=1.0):
        """Constructor

        Args:
            knowledge (Knowledge): Knowledge base
            random_gen (numpy.random.Generator)
            mutation_rate (float): Mutation rate
        """
        super().__init__()
        self.__mutation_rate = mutation_rate
        self.__knowledge = knowledge
        self._random_gen = random_gen

    def _do(self, problem, X, **kwargs):
        """Run the mutation on the given population

        Args:
            problem (Problem): Optimization problem (e.g. PittsburghProblem)
            X (object[,]): Population. The shape is (pop_size, n_var)
            **kwargs (dict): Other arguments taken by Pymoo crossover object

        Returns:
             X (object[,]): Population after mutation
        """
        if len(X) == 0:
            return X
        # for each individual
        cdef PittsburghSolution sol = X[0, 0]
        cdef MichiganSolution sol_m = sol.get_var(0)
        cdef Dataset training_set = sol_m.get_rule_builder().get_training_dataset()
        training_set_size = training_set.get_size()
        dim = training_set.get_num_dim()

        for s in range(len(X)):
            for i in range(len(X[s])):
                # for each michigan solution (rule)
                for michigan_sol_i in range(X[s][i].get_num_vars()):
                    mutated_dim = self._random_gen.integers(0, dim)

                    # Check if the mutated dim is categorical (<0) or numerical (>=0)
                    var_of_random_pattern = (training_set.get_pattern(self._random_gen.integers(0, training_set_size))
                                             .get_attribute_value(mutated_dim))

                    if var_of_random_pattern >= 0:
                        num_candidate_values = self.__knowledge.get_num_fuzzy_sets(mutated_dim)
                        if num_candidate_values <= 1:
                            break  # Only one possible value so we can't change it

                        current_michigan_solution = X[s][i].get_var(michigan_sol_i)
                        # print(mutated_dim, current_michigan_solution.get_num_vars(), current_michigan_solution)
                        current_fuzzy_set_index = current_michigan_solution.get_var(mutated_dim)
                        new_fuzzy_set_index = self._random_gen.integers(0, num_candidate_values - 1)

                        # Prevent the value from staying the same
                        if new_fuzzy_set_index >= current_fuzzy_set_index:
                            new_fuzzy_set_index += 1

                        new_michigan_solution = copy.deepcopy(current_michigan_solution)
                        new_michigan_solution.set_var(mutated_dim, new_fuzzy_set_index)
                        new_michigan_solution.learning()

                        if not new_michigan_solution.get_consequent().is_rejected():
                            X[s][i].set_var(michigan_sol_i, new_michigan_solution)
                    else:
                        current_michigan_solution = X[s][i].get_var(michigan_sol_i)
                        current_var_value = current_michigan_solution.get_var(mutated_dim)

                        current_michigan_solution.set_var(mutated_dim, round(var_of_random_pattern))
                        current_michigan_solution.learning()

                        if current_michigan_solution.get_rule().is_rejected_class_label():
                            current_michigan_solution.set_var(mutated_dim, current_var_value)
                            current_michigan_solution.learning()


        return X