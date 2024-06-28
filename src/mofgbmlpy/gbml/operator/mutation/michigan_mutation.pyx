from pymoo.core.mutation import Mutation

from mofgbmlpy.fuzzy.knowledge.knowledge import Knowledge
from mofgbmlpy.fuzzy.rule.consequent.learning.learning_basic import LearningBasic
import random

from mofgbmlpy.gbml.solution.michigan_solution import MichiganSolution


class MichiganMutation(Mutation):
    __learner = None
    __knowledge = None

    def __init__(self, training_set, knowledge, mutation_rt):
        super().__init__()
        self.__learner = LearningBasic(training_set)
        self.__knowledge = knowledge
        self.__mutation_rt = mutation_rt

    def _do(self, problem, X, **kwargs):
        # for each individual
        for i in range(len(X)):
            cdef MichiganSolution sol = X[i, 0]
            # for each var
            indices = X[i, 0].get_antecedent().get_antecedent_indices()
            for j in range(sol.get_num_vars()):
                if random.rand() > self.__mutation_rt:
                    continue

                # TODO: check Java version for categorical attributes

                num_fuzzy_sets = self.__knowledge.get_num_fuzzy_sets(j) # TODO: check if _vars and antecedent indices are the same array object
                indices[j] = random.randint(num_fuzzy_sets - 1)
            X[i, 0].set_consequent(self.__learner.learning(X[i, 0].get_antecedent()))
        return X
