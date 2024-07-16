import copy
import time

from pymoo.core.mutation import Mutation

from mofgbmlpy.fuzzy.knowledge.knowledge import Knowledge
from mofgbmlpy.gbml.solution import michigan_solution
from mofgbmlpy.fuzzy.rule.consequent.learning.learning_basic import LearningBasic
import random


class PittsburghMutation(Mutation):
    __learner = None
    __mutation_rate = None
    __knowledge = None

    def __init__(self, training_set, knowledge, mutation_rate=1.0):
        super().__init__()
        self.__learner = LearningBasic(training_set)
        self.__mutation_rate = mutation_rate
        self.__knowledge = knowledge

    def _do(self, problem, X, **kwargs):
        # for each individual
        training_set = self.__learner.get_training_set()
        training_set_size = training_set.get_size()
        dim = training_set.get_num_dim()

        for i in range(len(X[0])):
            # for each michigan solution (rule)
            for michigan_sol_i in range(X[0][i].get_num_vars()):
                mutated_dim = random.randint(0, dim-1)

                # Check if the mutated dim is categorical (<0) or numerical (>=0)
                var_of_random_pattern = (training_set.get_pattern(random.randint(0, training_set_size-1))
                                         .get_attribute_value(mutated_dim))

                if var_of_random_pattern >= 0:
                    num_candidate_values = self.__knowledge.get_num_fuzzy_sets(mutated_dim)
                    if num_candidate_values <= 1:
                        break  # Only one possible value so we can't change it

                    current_michigan_solution = X[0][i].get_var(michigan_sol_i)
                    # print(mutated_dim, current_michigan_solution.get_num_vars(), current_michigan_solution)
                    current_fuzzy_set_id = current_michigan_solution.get_var(mutated_dim)
                    new_fuzzy_set_id = random.randint(0, num_candidate_values - 2)

                    # Prevent the value from staying the same
                    if new_fuzzy_set_id >= current_fuzzy_set_id:
                        new_fuzzy_set_id += 1

                    new_michigan_solution = copy.deepcopy(current_michigan_solution)
                    new_michigan_solution.set_var(mutated_dim, new_fuzzy_set_id)
                    new_michigan_solution.learning()

                    if not new_michigan_solution.get_consequent().is_rejected():
                        X[0][i].set_var(michigan_sol_i, new_michigan_solution)
                else:
                    current_michigan_solution = X[0][i].get_var(michigan_sol_i)
                    current_var_value = current_michigan_solution.get_var(mutated_dim)

                    current_michigan_solution.set_var(mutated_dim, round(var_of_random_pattern))
                    current_michigan_solution.learning()

                    if current_michigan_solution.get_rule().is_rejected_class_label():
                        current_michigan_solution.set_var(mutated_dim, current_var_value)
                        current_michigan_solution.learning()


        return X