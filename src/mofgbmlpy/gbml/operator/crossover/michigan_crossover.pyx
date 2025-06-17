import copy

from functools import cmp_to_key
import numpy as np
from mofgbmlpy.gbml.operator.crossover.pymoo_deepcopy_crossover import PymooDeepcopyCrossover
from pymoo.core.crossover import Crossover
from pymoo.core.population import Population

from mofgbmlpy.gbml.operator.crossover.uniform_crossover_single_offspring_michigan import UniformCrossoverSingleOffspringMichigan
from mofgbmlpy.gbml.operator.mutation.michigan_mutation import MichiganMutation
from mofgbmlpy.gbml.operator.selection.nary_tournament_selection_on_fitness import NaryTournamentSelectionOnFitness
from mofgbmlpy.gbml.operator.survival.rule_style_survival import RuleStyleSurvival
from mofgbmlpy.gbml.problem.michigan_problem import MichiganProblem


class MichiganCrossover(PymooDeepcopyCrossover):
    """Apply the Michigan crossover on the Michigan solutions of one Pittsburgh solution

    Attributes:
        __crossover_rate (float): Probability that a crossover occurs
        __rule_change_rate (float): Ratio of rules that will be changed in the parent (the Pittsburgh solution)
        __training_set (Dataset): Training dataset
        __knowledge (Knowledge): Knowledge base
        __max_num_rules (int): Max number of rules that the Pittsburgh solution can contain
        _random_gen (numpy.random.Generator): Random generator
    """

    def __init__(self, rule_change_rate, training_set, knowledge, max_num_rules, random_gen, prob=1, **kwargs):
        """Constructor

        Args:
            rule_change_rate (float): Ratio of rules that will be changed in the parent (the Pittsburgh solution)
            training_set (Dataset): Training dataset
            knowledge (Knowledge): Knowledge base
            max_num_rules (int): Max number of rules that the Pittsburgh solution can contain
            random_gen (numpy.random.Generator): Random generator
            prob (float): Probability that a crossover occurs
        """
        super().__init__(n_parents=1, n_offsprings=1, random_gen=random_gen, prob=prob, **kwargs)
        self.__crossover_rate = prob
        self.__rule_change_rate = rule_change_rate
        self.__training_set = training_set
        self.__knowledge = knowledge
        self.__max_num_rules = max_num_rules
        self._random_gen = random_gen

    def ga_rules_gen(self, crossover, mutation, selection, pop, problem, num_offspring, n_parents):
        """Generate rules using a genetic algorithm

        Args:
            crossover (Crossover): Crossover operator object
            mutation (Mutation): Mutation operator object
            selection (Selection): Selection operator object (select the mating pool in a population)
            pop (Population): Population
            problem (Problem): Optimization problem definition
            num_offspring (int): Number of offspring (rules) to be generated
            n_parents (int): Number of parents used to generate the rules

        Returns:
            list: Generated rules
        """
        mating_pop = selection.do(problem, pop, num_offspring, n_parents, to_pop=False)
        generated_solutions = []

        for i in range(0, num_offspring):
            parents = mating_pop[i]
            p1_obj = pop[parents[0]].X[0]
            p2_obj = pop[parents[1]].X[0]

            offspring = crossover.do(problem, pop, parents=[parents])
            offspring = mutation.do(problem, offspring)

            for j in range(len(offspring)):
                offspring[j].X[0].learning()

                if offspring[j].X[0].get_rule().is_rejected_class_label():
                    generated_solutions.append(copy.deepcopy(p1_obj))
                    if len(generated_solutions) == num_offspring:
                        return generated_solutions
                    generated_solutions.append(copy.deepcopy(p2_obj))
                else:
                    generated_solutions.append(offspring[j].X[0])

                if len(generated_solutions) == num_offspring:
                    # return generated_solutions # TODO: recheck the Java version, since it seems the Java code does not always return num_ga
                    break

        return generated_solutions

    @staticmethod
    def radix_sort_michigan(x, y):
        """Radix sort Michigan solutions based on their antecedent indices

        Args:
            x (MichiganSolution): First Michigan solution
            y (MichiganSolution): Second Michigan solution

        Returns:
            int: Comparison result
        """
        for i in range(x.get_num_vars()):
            x_var = x.get_var(i)
            y_var = y.get_var(i)

            if x_var < y_var:
                return -1
            elif x_var > y_var:
                return 1
        return 0

    def heuristic_rules_gen(self, parent, num_heuristic):
        generated_solutions = np.empty(num_heuristic, dtype=object)
        error_patterns = parent.get_errored_patterns()
        lack_size = num_heuristic - len(error_patterns)

        if lack_size > 0:
            new_patterns = self._random_gen.choice(self.__training_set.get_patterns(), lack_size)
            error_patterns = np.concatenate((error_patterns, new_patterns))
        selected_error_patterns_indices = self._random_gen.choice(np.arange(len(error_patterns)),
                                                                  num_heuristic,
                                                                  replace=False)

        for j in range(num_heuristic):
            generated_solutions[j] = parent.get_michigan_solution_builder().create(pattern=error_patterns[selected_error_patterns_indices[j]])[0]
        return generated_solutions

    def _do(self, problem, X, **kwargs):
        """Run the crossover on the given population

        Args:
            problem (Problem): Optimization problem (e.g. PittsburghProblem)
            X (object[,]): Population. The shape is (1, n_matings, n_var),
            **kwargs (dict): Other arguments taken by Pymoo crossover object

        Returns:
            float[,,]: Crossover offspring. Shape: (1, n_matings, 1)
        """
        # Note: X contains Pittsburgh solutions
        _, n_matings, n_var = X.shape
        Y = np.zeros((1, n_matings, 1), dtype=object)

        num_dim = X[0, 0, 0].get_var(0).get_num_vars()

        for i in range(n_matings):
            generated_solutions = []

            parent = X[0, i, 0]

            # 1. Calculate number of all of generating rules

            num_rules_on_parent = parent.get_num_vars()
            num_generating_rules = int(self.__rule_change_rate * num_rules_on_parent + 1)

            # 2. Calculate number of rules generated by GA and Heuristic rule generation method

            if num_generating_rules % 2 == 0:
                num_heuristic = num_generating_rules // 2
            else:
                num_heuristic = (num_generating_rules - 1) // 2 + self._random_gen.integers(0, 2)

            # 3. Heuristic Rule Generation

            if num_heuristic > 0:
                generated_solutions = self.heuristic_rules_gen(parent, num_heuristic)

            # 4. Rule Generation by Genetic Algorithm - Michigan-style GA
            num_ga = num_generating_rules - num_heuristic

            if num_ga > 0:
                michigan_problem = MichiganProblem([], # Objectives are not used
                                                   problem.get_num_constraints(),
                                                   problem.get_training_set(),
                                                   problem.get_rule_builder())

                crossover = UniformCrossoverSingleOffspringMichigan(self._random_gen, self.__crossover_rate)

                mutation_rt = 1/self.__training_set.get_num_dim()
                mutation = MichiganMutation(self.__knowledge, mutation_rt, self._random_gen)

                if parent.get_num_vars() == 1:
                    # no crossover
                    tournament_size = 1
                else:
                    tournament_size = 2
                selection = NaryTournamentSelectionOnFitness(tournament_size)

                michigan_solutions_array = np.empty((parent.get_num_vars(), 1), dtype=object)
                parent_vars = parent.get_vars()
                for j in range(michigan_solutions_array.shape[0]):
                    michigan_solutions_array[j, 0] = parent_vars[j]
                michigan_population = Population.new(X=michigan_solutions_array)

                ga_generated_solutions = self.ga_rules_gen(crossover,
                                                            mutation,
                                                            selection,
                                                            michigan_population,
                                                            michigan_problem,
                                                            num_ga,
                                                            2)

                generated_solutions = np.concatenate((generated_solutions, ga_generated_solutions))

            # 5. Replacement: Single objective maximization replacement based on the fitness value
            generated_solutions = RuleStyleSurvival.replace(parent.get_vars(), generated_solutions, self.__max_num_rules)

            # generated_solutions = np.array(sorted(
            #     generated_solutions,
            #     key=cmp_to_key(lambda x, y: MichiganCrossover.radix_sort_michigan(x, y))
            # ))

            offspring = copy.deepcopy(parent)
            offspring.clear_vars()
            offspring.clear_attributes()
            offspring.set_vars(generated_solutions)

            Y[0, i, 0] = offspring
        return Y

    def execute(self, problem, X, **kwargs):
        """Public version of the _do function (needed for the Hybrid crossover). Run the crossover on the given population

        Args:
            problem (Problem): Optimization problem (e.g. PittsburghProblem)
            X (object[,]): Population. The shape is (n_matings, n_var),
            **kwargs (dict): Other arguments taken by Pymoo crossover object

        Returns:
            float[,,]: Crossover offspring. Shape: (1, n_matings, 1)
        """
        return self._do(problem, X, **kwargs)
