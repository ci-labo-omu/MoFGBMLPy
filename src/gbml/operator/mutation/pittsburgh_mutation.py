import copy

from pymoo.core.mutation import Mutation

from fuzzy.knowledge.knowledge import Knowledge
from src.fuzzy.rule.consequent.learning.learning_basic import LearningBasic
import random


class PittsburghMutation(Mutation):
    __learner = None
    __mutation_rate = None

    def __init__(self, training_set, mutation_rate=1.0):
        super().__init__()
        self.__learner = LearningBasic(training_set)
        self.__mutation_rate = mutation_rate

    def _do(self, problem, X, **kwargs):
        # for each individual
        dim = self.__learner.get_training_set().get_num_dim()
        knowledge = Knowledge.get_instance()

        for i in range(len(X[0])):
            # for each michigan solution (rule)
            for michigan_sol_i in range(X[0][i].get_num_vars()):
                mutated_dim = random.randint(0, dim-1)
                num_candidate_values = knowledge.get_num_fuzzy_sets(mutated_dim)
                if num_candidate_values <= 1:
                    break  # Only one possible value so we can't change it

                current_michigan_solution = X[0][i].get_var(michigan_sol_i)
                current_fuzzy_set_id = current_michigan_solution.get_var(dim)
                new_fuzzy_set_id = random.randint(0, num_candidate_values - 2)

                # Prevent the value from staying the same
                if new_fuzzy_set_id >= current_fuzzy_set_id:
                    new_fuzzy_set_id += 1

                new_michigan_solution = copy.copy(current_michigan_solution)
                new_michigan_solution.set_var(mutated_dim, new_fuzzy_set_id)
                new_michigan_solution.learning()

                if not new_michigan_solution.get_consequent().is_rejected():
                    X[0][i].set_var(michigan_sol_i, new_michigan_solution)

        return X