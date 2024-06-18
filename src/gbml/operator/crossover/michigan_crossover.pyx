# import copy
# import random
# import time
#
# import numpy as np
# from pymoo.algorithms.soo.nonconvex.ga import GA
# from pymoo.core.crossover import Crossover
#
# from gbml.problem.michigan_problem import MichiganProblem
# from gbml.solution.michigan_solution import MichiganSolution
#
#
# class MichiganCrossover(Crossover):
#     __rule_change_rate = None
#     __training_set = None
#
#     def __init__(self, rule_change_rate, training_set, prob=0.9):
#         super().__init__(1, 1, prob)
#         self.__rule_change_rate = rule_change_rate
#         self.__training_set = training_set
#
#     def _do(self, problem, X, **kwargs):
#         _, n_matings, n_var = X.shape
#         Y = np.zeros((1, n_matings, 1), dtype=object)
#
#         num_dim = X[0, 0, 0].get_var(0).get_num_vars()
#
#         for i in range(n_matings):
#             generated_solutions = []
#
#             parent = X[0, i, 0]
#
#             # 1. Calculate number of all of generating rules
#
#             num_rules_on_parent = parent.get_num_vars()
#             num_generating_rules = int(self.__rule_change_rate * num_rules_on_parent + 1)
#
#             # 2. Calculate numbers of rules generated by GA and Heuristic rule generation method
#
#             if num_generating_rules % 2 == 0:
#                 num_heuristic = num_generating_rules // 2
#             else:
#                 num_heuristic = (num_generating_rules - 1) // 2 + random.randint(0, 1)
#
#             # 3. Heuristic Rule Generation
#
#             if num_heuristic > 0:
#                 error_patterns = parent.get_errored_patterns()
#                 lack_size = num_heuristic - len(error_patterns)
#
#                 new_patterns = np.random.choice(self.__training_set.get_patterns(), lack_size)
#                 error_patterns = np.concatenate(error_patterns, new_patterns)
#                 selected_error_patterns = np.random.choice(error_patterns, num_heuristic, replace=False)
#
#                 for j in range(num_heuristic):
#                     generated_solutions.append(
#                         parent.get_michigan_solution_builder().create(pattern=selected_error_patterns[j]))
#
#             # 4. Rule Generation by Genetic Algorithm - Michigan-style GA
#             num_ga = num_generating_rules - num_heuristic
#
#             if num_ga > 0:
#                 michigan_problem = MichiganProblem(num_dim,
#                                                    problem.get_num_objectives(),
#                                                    problem.get_num_constraints(),
#                                                    problem.get_training_set(),
#                                                    problem.get_rule_builder())
#
#                 selection = NaryTournamentSelection()
#
#                 X = np.full((5000, 1), 0.5)
#                 pop = Population.new(X=X)
#                 rules_parents = selection.do(parent.get_vars())
#
#             #TODO
#             raise Exception("Not yet implemented")
#
#             # Y[0, i] = copy.copy(p1)
#             # Y[0, i, 0].clear_vars()
#             # Y[0, i, 0].clear_attributes()
#             #
#             # num_rules_from_p1, num_rules_from_p2 = self.get_num_rules_from_parents(p1.get_num_vars(), p2.get_num_vars(),
#             #                                                                        n_var)
#             # rules_idx_from_p1 = np.random.choice(list(range(p1.get_num_vars())), num_rules_from_p1, replace=False)
#             # rules_idx_from_p2 = np.random.choice(list(range(p2.get_num_vars())), num_rules_from_p2, replace=False)
#             #
#             # for rule_idx in rules_idx_from_p1:
#             #     Y[0, i, 0].add_var(copy.copy(p1.get_var(rule_idx)))
#             # for rule_idx in rules_idx_from_p2:
#             #     Y[0, i, 0].add_var(copy.copy(p2.get_var(rule_idx)))
#
#         return Y
#
#     def execute(self, problem, X, **kwargs):
#         return self._do(problem, X, **kwargs)
