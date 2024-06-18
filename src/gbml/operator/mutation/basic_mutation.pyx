from pymoo.core.mutation import Mutation

from fuzzy.knowledge.knowledge import Knowledge
from fuzzy.rule.consequent.learning.learning_basic import LearningBasic
import random


class BasicMutation(Mutation):
    __learner = None
    __knowledge = None

    def __init__(self, training_set, knowledge):
        super().__init__()
        self.__learner = LearningBasic(training_set)
        self.__knowledge = knowledge

    def _do(self, problem, X, **kwargs):
        # for each individual
        for i in range(len(X)):
            indices = X[i, 0].get_antecedent().get_antecedent_indices()
            for j in range(len(indices)):
                num_fuzzy_sets = self.__knowledge.get_num_fuzzy_sets(j)
                indices[j] = random.randint(0, num_fuzzy_sets - 1)

            X[i, 0].set_consequent(self.__learner.learning(X[i, 0].get_antecedent()))
        return X
