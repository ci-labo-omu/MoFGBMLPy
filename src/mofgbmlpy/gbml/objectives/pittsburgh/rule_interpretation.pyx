# from mofgbmlpy.gbml.solution.michigan_solution import MichiganSolution
# from mofgbmlpy.gbml.solution.pittsburgh_solution import PittsburghSolution
#
# cpdef num_rules(solution):
#     if isinstance(solution, PittsburghSolution):
#         knowledge = solution.get_michigan_solution_builder().get_rule_builder().get_knowledge()
#         cpdef double rule_interpretation = 0
#         for i in range(solution.get_num_vars()):
#             cdef MichiganSolution michigan_sol = solution.get_var(i)
#             if michigan_sol.get_num_wins() > 0:
#                 rule_interpretation += 1
#
#             rule_interpretation += michigan_sol.get_length()*1e-4
#
#             indices = michigan_sol.get_antecedent().get_antecedent_indices()
#             for j in range(len(indices)):
#                 if knowledge.get_fuzzy_set(j, indices[j]).get_division_type() == "entropy_division":
#                     rule_interpretation +=1e-8
#         return rule_interpretation
#     else:
#         raise Exception("Solution must be of type PittsburghSolution")