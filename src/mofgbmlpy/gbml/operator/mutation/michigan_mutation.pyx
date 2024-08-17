from pymoo.core.mutation import Mutation

from mofgbmlpy.data.dataset cimport Dataset
from mofgbmlpy.fuzzy.knowledge.knowledge import Knowledge
from mofgbmlpy.fuzzy.rule.consequent.learning.learning_basic import LearningBasic

from mofgbmlpy.gbml.solution.michigan_solution cimport MichiganSolution


class MichiganMutation(Mutation):
    """Michigan mutation operator

    Attributes:
        __knowledge (Knowledge): Knowledge base
        __mutation_rt (float): Mutation rate
        _random_gen (numpy.random.Generator):
    """

    def __init__(self, knowledge, mutation_rt, random_gen):
        super().__init__()
        self.__knowledge = knowledge
        self.__mutation_rt = mutation_rt
        self._random_gen = random_gen

    def _do(self, problem, X, **kwargs):
        """Run the mutation on the given population

        Args:
            problem (Problem): Optimization problem (e.g. MichiganProblem)
            X (object[,]): Population. The shape is (pop_size, n_var)
            **kwargs (dict): Other arguments taken by Pymoo crossover object

        Returns:
             X (object[,]): Population after mutation
        """
        if len(X)==0:
            return X

        # for each individual
        cdef MichiganSolution sol
        sol = X[0, 0]
        cdef Dataset training_set = sol.get_rule_builder().get_training_dataset()
        cdef int training_set_size = training_set.get_size()

        for i in range(len(X)):
            sol = X[i, 0]
            # for each var
            indices = sol.get_antecedent().get_antecedent_indices()
            for j in range(sol.get_num_vars()):
                if self._random_gen.random() > self.__mutation_rt:
                    continue

                # Check if the mutated dim is categorical (<0) or numerical (>=0)
                var_of_random_pattern = (training_set.get_pattern(self._random_gen.integers(0, training_set_size))
                                         .get_attribute_value(j))

                if var_of_random_pattern >= 0:
                    num_fuzzy_sets = self.__knowledge.get_num_fuzzy_sets(j)
                    new_fuzzy_set = self._random_gen.integers(0, num_fuzzy_sets - 1)

                    # To avoid getting the same value again we do the following
                    if new_fuzzy_set < indices[j]:
                        indices[j] = new_fuzzy_set
                    else:
                        indices[j] = new_fuzzy_set + 1
                else:
                    # Categorical attribute
                    indices[j] = round(var_of_random_pattern)
            sol.learning()
        return X
