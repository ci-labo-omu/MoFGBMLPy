from pymoo.core.mutation import Mutation

from mofgbmlpy.data.dataset cimport Dataset
from mofgbmlpy.fuzzy.knowledge.knowledge import Knowledge
from mofgbmlpy.fuzzy.rule.consequent.learning.learning_basic import LearningBasic

from mofgbmlpy.gbml.solution.michigan_solution cimport MichiganSolution


class MichiganMutation(Mutation):
    __learner = None
    __knowledge = None
    __random_gen = None

    def __init__(self, training_set, knowledge, mutation_rt, random_gen):
        super().__init__()
        self.__learner = LearningBasic(training_set)
        self.__knowledge = knowledge
        self.__mutation_rt = mutation_rt
        self._random_gen = random_gen

    def _do(self, problem, X, **kwargs):
        # for each individual
        cdef MichiganSolution sol

        cdef Dataset training_set = self.__learner.get_training_set()
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
