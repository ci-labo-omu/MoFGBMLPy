import copy
import random
import time

import numpy as np
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.crossover import Crossover
from pymoo.core.population import Population

from mofgbmlpy.gbml.operator.crossover.uniform_crossover_single_offspring import UniformCrossoverSingleOffspring
from mofgbmlpy.gbml.operator.mutation.michigan_mutation import MichiganMutation
from mofgbmlpy.gbml.operator.selection.NaryTournamentSelectionOnFitness import NaryTournamentSelectionOnFitness
from mofgbmlpy.gbml.operator.survival.rule_style_survival import RuleStyleSurvival
from mofgbmlpy.gbml.problem.michigan_problem import MichiganProblem
from mofgbmlpy.gbml.solution.michigan_solution import MichiganSolution
from mofgbmlpy.gbml.solution.pittsburgh_solution import PittsburghSolution


class MichiganCrossover(Crossover):
    __rule_change_rate = None
    __training_set = None
    __knowledge = None
    __prob_float = None

    def __init__(self, rule_change_rate, training_set, knowledge, max_num_rules, prob=0.9):
        super().__init__(1, 1, prob)
        self.__prob_float = prob
        self.__rule_change_rate = rule_change_rate
        self.__training_set = training_set
        self.__knowledge = knowledge
        self.__max_num_rules = max_num_rules

    def ga_rules_gen(self, crossover, mutation, selection, pop, problem, mating_pool_size, n_parents, num_ga):
        mating_pop = selection.do(problem, pop, mating_pool_size, n_parents, to_pop=False)
        generated_solutions = []

        for i in range(0, mating_pool_size, 2):
            parents = mating_pop[i]
            p1_obj = pop[parents[0]].X[0]
            p2_obj = pop[parents[1]].X[0]

            print("P1",np.copy(p1_obj.get_vars()))
            ind = p1_obj.get_rule().get_antecedent().get_antecedent_indices()
            print(np.copy(ind))

            print("P2",np.copy(p2_obj.get_vars()))
            ind = p2_obj.get_rule().get_antecedent().get_antecedent_indices()
            print(np.copy(ind))
            print("###")

            # TODO: remove these "if" tests
            if p1_obj.get_rule().is_rejected_class_label():
                print("P exc = ", p1_obj)
                raise Exception(f"Invalid parent A")
            if p2_obj.get_rule().is_rejected_class_label():
                print("P exc = ", p2_obj)
                raise Exception(f"Invalid parent B")

            offspring = crossover.do(problem, pop, parents=[parents])
            if p1_obj.get_rule().is_rejected_class_label():
                print("P exc = ", p1_obj)
                raise Exception(f"Invalid parent A")
            if p2_obj.get_rule().is_rejected_class_label():
                print("P exc = ", p2_obj)
                raise Exception(f"Invalid parent B")

            offspring = mutation.do(problem, offspring)

            if p1_obj.get_rule().is_rejected_class_label():
                print("P exc = ", p1_obj)
                raise Exception(f"Invalid parent A")
            if p2_obj.get_rule().is_rejected_class_label():
                print("P exc = ", p2_obj)
                raise Exception(f"Invalid parent B")

            for j in range(len(offspring)):
                if offspring[j].X[0].get_rule().is_rejected_class_label():
                    generated_solutions.append(copy.deepcopy(p1_obj))
                    if len(generated_solutions) == num_ga:
                        return generated_solutions
                    generated_solutions.append(copy.deepcopy(p2_obj))
                else:
                    generated_solutions.append(offspring[j].X[0])
                if len(generated_solutions) == num_ga:
                    return generated_solutions

            if p1_obj.get_rule().is_rejected_class_label():
                print("P exc = ", p1_obj)
                raise Exception(f"Invalid parent A")
            if p2_obj.get_rule().is_rejected_class_label():
                print("P exc = ", p2_obj)
                raise Exception(f"Invalid parent B")

        return generated_solutions

    def _do(self, problem, X, **kwargs):
        # Note: X contains Pittsburgh solutions
        n_matings, n_var = X.shape
        Y = np.zeros((1, n_matings, 1), dtype=object)

        num_dim = X[0, 0].get_var(0).get_num_vars()

        for sol in X:
            for m in sol[0].get_vars():
                # print(m)
                if m.get_rule().is_rejected_class_label():
                    raise Exception("Invalid parent")

        # print("P19", X[19, 0])

        for i in range(n_matings):
            generated_solutions = []

            parent = X[i, 0]
            for m in parent.get_vars():
                # print(m)
                if m.get_rule().is_rejected_class_label():
                    print("P exc = ", parent)
                    raise Exception(f"Invalid parent {i} var {m}")

            # 1. Calculate number of all of generating rules

            num_rules_on_parent = parent.get_num_vars()
            num_generating_rules = int(self.__rule_change_rate * num_rules_on_parent + 1)

            # 2. Calculate numbers of rules generated by GA and Heuristic rule generation method

            if num_generating_rules % 2 == 0:
                num_heuristic = num_generating_rules // 2
            else:
                num_heuristic = (num_generating_rules - 1) // 2 + random.randint(0, 1)

            # 3. Heuristic Rule Generation

            if num_heuristic > 0:
                error_patterns = parent.get_errored_patterns()
                lack_size = num_heuristic - len(error_patterns)

                if lack_size > 0:
                    new_patterns = np.random.choice(self.__training_set.get_patterns(), lack_size)
                    error_patterns = np.concatenate(error_patterns, new_patterns)
                selected_error_patterns = np.random.choice(error_patterns, num_heuristic, replace=False)

                for j in range(num_heuristic):
                    generated_solutions.append(
                        parent.get_michigan_solution_builder().create(pattern=selected_error_patterns[j])[0])

            for m in parent.get_vars():
                # print(m)
                if m.get_rule().is_rejected_class_label():
                    print("P exc = ", parent)
                    raise Exception(f"Invalid parent {i} var {m}")
            # 4. Rule Generation by Genetic Algorithm - Michigan-style GA
            num_ga = num_generating_rules - num_heuristic

            if num_ga > 0:
                michigan_problem = MichiganProblem(num_dim,
                                                   problem.get_num_objectives(),
                                                   problem.get_num_constraints(),
                                                   problem.get_training_set(),
                                                   problem.get_rule_builder())

                crossover = UniformCrossoverSingleOffspring(self.__prob_float)

                mutation_rt = 1/self.__training_set.get_num_dim()
                mutation = MichiganMutation(self.__training_set, self.__knowledge, mutation_rt)

                if parent.get_num_vars() == 1:
                    tournament_size = 1
                else:
                    tournament_size = 2
                mating_pool_size = num_ga * crossover.n_parents // crossover.n_offsprings
                selection = NaryTournamentSelectionOnFitness(tournament_size)


                for m in parent.get_vars():
                    # print(m)
                    if m.get_rule().is_rejected_class_label():
                        print("P exc = ", parent)
                        raise Exception(f"Invalid parent {i} var {m}")

                michigan_solutions_array = parent.get_vars().reshape((parent.get_num_vars(), 1))
                michigan_population = Population.new(X=michigan_solutions_array)

                ga_generated_solutions = self.ga_rules_gen(crossover,
                                                            mutation,
                                                            selection,
                                                            michigan_population,
                                                            problem,
                                                            mating_pool_size,
                                                            2,
                                                            num_ga)

                generated_solutions = np.concatenate((generated_solutions, ga_generated_solutions))
                for m in parent.get_vars():
                    # print(m)
                    if m.get_rule().is_rejected_class_label():
                        print("P exc = ", parent)
                        raise Exception(f"Invalid parent {i} var {m}")

            # 5. Replacement: Single objective maximization replacement
            generated_solutions = RuleStyleSurvival.replace(parent.get_vars(), generated_solutions, self.__max_num_rules)

            offspring = copy.deepcopy(parent)
            offspring.clear_vars()
            offspring.set_vars(generated_solutions)

            for m in parent.get_vars():
                # print(m)
                if m.get_rule().is_rejected_class_label():
                    print("P exc = ", parent)
                    raise Exception(f"Invalid parent {i} var {m}")

            Y[0, i, 0] = offspring
        return Y

    def execute(self, problem, X, **kwargs):
        return self._do(problem, X, **kwargs)
