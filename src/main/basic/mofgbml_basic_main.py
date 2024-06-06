from pymoo.core.repair import Repair
from pymoo.core.survival import Survival
from pymoo.visualization.scatter import Scatter

from src.gbml.solution.pittsburgh_solution import PittsburghSolution
from fuzzy.rule.antecedent.factory.heuristic_antecedent_factory import HeuristicAntecedentFactory
from gbml.solution.michigan_solution import MichiganSolution
from src.fuzzy.classifier.classification.single_winner_rule_selection import SingleWinnerRuleSelection
from src.fuzzy.classifier.classifier import Classifier
from src.fuzzy.rule.consequent.consequent import Consequent
from src.fuzzy.fuzzy_term.fuzzy_term_triangular import FuzzyTermTriangular
from src.main.experience_parameter import ExperienceParameter
from src.data.input import Input
from src.utility.Output import Output
from src.main.consts import Consts
from src.main.basic.mofgbml_basic_args import MoFGBMLBasicArgs
from src.data.dataset_manager import DataSetManager
import sys
import os
from pymoo.core.problem import Problem
from pymoo.core.duplicate import ElementwiseDuplicateElimination
from pymoo.core.selection import Selection
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.termination import Termination
from pymoo.core.sampling import Sampling
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from src.fuzzy.knowledge.knowledge import Knowledge
import random
import numpy as np

from src.fuzzy.rule.consequent.learning.learning_basic import LearningBasic
from src.fuzzy.rule.rule_basic import RuleBasic
from src.fuzzy.rule.antecedent.antecedent import Antecedent
from src.fuzzy.rule.antecedent.factory.all_combination_antecedent_factory import AllCombinationAntecedentFactory
from src.fuzzy.knowledge.homo_triangle_knowledge_factory import HomoTriangleKnowledgeFactory


class MoFGBMLBasicMain:
    @staticmethod
    def main(args):
        # TODO: print information

        # Consts.set()
        #...

        Output.mkdirs(Consts.ROOTFOLDER)

        # set command arguments to static variables
        MoFGBMLBasicArgs.load(args)

        # Save const params
        file_name = str(os.path.join(Consts.EXPERIMENT_ID_DIR, "Consts.txt"))
        Output.writeln(file_name, str(Consts()), True)
        Output.writeln(file_name, str(MoFGBMLBasicArgs()), True)

        # Load dataset
        ExperienceParameter.get_instance().set_class_label_type(ExperienceParameter.ClassLabelType.SINGLE)
        Input.load_train_test_files(MoFGBMLBasicArgs.get_train_file(), MoFGBMLBasicArgs.get_test_file())
        manager = DataSetManager.get_instance()
        test = manager.get_tests()[0]
        train = manager.get_trains()[0]

        # Run the algo
        MoFGBMLBasicMain.hybrid_style_mofgbml(train, test)

    class BasicProblem(Problem):
        __winner_solution_for_each_pattern = None
        __num_vars_pittsburgh = None
        __num_objectives_pittsburgh = None
        __num_constraints_pittsburgh = None
        __training_ds = None
        __michigan_solution_builder = None
        __classifier = None
        __num_vars = 0

        class WinnerSolution:
            __max_fitness_value = None
            __solution_index = None

            def __init__(self, max_fitness_value=None, __solution_index=None):
                self.__max_fitness_value = max_fitness_value
                self.__solution_index = __solution_index

            def get_max_fitness_value(self):
                return self.__max_fitness_value

            def get_solution_index(self):
                return self.__solution_index

        def __init__(self,
                     num_vars,
                     num_objectives,
                     num_constraints,
                     training_dataset,
                     michigan_solution_builder,
                     classifier):

            super().__init__(n_var=num_vars, n_obj=num_objectives)
            self.__training_ds = training_dataset
            self.__winner_solution_for_each_pattern = np.array([MoFGBMLBasicMain.BasicProblem.WinnerSolution()]*training_dataset.get_size(), dtype=object)
            self.__num_vars = num_vars
            self.__michigan_solution_builder = michigan_solution_builder
            self.__classifier = classifier

        def create_solution(self):
            pittsburgh_solution = PittsburghSolution(self.__num_vars,
                                          self.__num_objectives_pittsburgh,
                                          self.__num_constraints_pittsburgh,
                                          self.__michigan_solution_builder.copy(),
                                          self.__classifier.copy())

            michigan_solutions = self.__michigan_solution_builder.create_michigan_solution(self.__num_vars)
            for solution in michigan_solutions:
                pittsburgh_solution.add_var(solution)
                
            return pittsburgh_solution

        def _evaluate(self, X, out, *args, **kwargs):
            out["F"] = np.zeros((len(X), 2), dtype=np.float_)
            patterns = self.__training_ds.get_patterns()

            for i in range(len(X)):
                out["F"][i][0] = 0
                out["F"][i][1] = X[i].get_rule_length()

                if X[i, 0].is_rejected_class_label():
                    out["F"][i][0] = -1  # This rule must be deleted during environmental selection
                    continue

                for j in range(len(patterns)):
                    fitness_value = X[i].get_fitness_value(patterns[j].get_attribute_vector())
                    max_fitness_value = self.__winner_solution_for_each_pattern[j].get_max_fitness_value()
                    if max_fitness_value is None or fitness_value > max_fitness_value:
                        self.__winner_solution_for_each_pattern[j] = MoFGBMLBasicMain.BasicProblem.WinnerSolution(-fitness_value, i)

            for i in range(len(patterns)):
                out["F"][self.__winner_solution_for_each_pattern[i].get_solution_index()][0] -= 1

    class BasicSampling(Sampling):
        __learner = None

        def __init__(self, training_set):
            super().__init__()
            self.__learner = LearningBasic(training_set)

        def _do(self, problem, n_samples, **kwargs):
            return np.array([problem.create_solution()]*n_samples, dtype=object)

    # class BasicSelection(Selection):
    #     def __init__(self, **kwargs) -> None:
    #         super().__init__(**kwargs)
    #
    #     def _do(self, problem, pop, n_select, n_parents, **kwargs):
    #         if len(pop) < 4:
    #             raise Exception("The population size must be greater or equal to 4")
    #         random_individuals = np.random.choice([i for i in range(len(pop))], size=2, replace=False)
    #
    #         return pop[random_individuals[0]], pop[random_individuals[1]]

    class BasicCrossover(Crossover):
        def __init__(self,
                     n_parents,
                     n_offsprings,
                     prob=0.9):
            super().__init__(n_parents, n_offsprings, prob)

        def _do(self, problem, X, **kwargs):
            _, n_matings, n_var = X.shape
            Y = np.zeros((1, n_matings, n_var), dtype=object)

            # for each mating provided
            for k in range(n_matings):
                # get the first and the second parent
                a, b = X[0, k, 0], X[1, k, 0]
                if random.random() < 0.5:
                    p1_antecedent_indices = X[0, k, 0].get_antecedent().get_antecedent_indices()
                    p2_antecedent_indices = X[1, k, 0].get_antecedent().get_antecedent_indices()

                    child_antecedent_indices = []
                    for i in range(len(p1_antecedent_indices)):
                        if random.random() < 0.5:
                            child_antecedent_indices.append(p2_antecedent_indices[i])
                        else:
                            child_antecedent_indices.append(p1_antecedent_indices[i])
                    Y[0, k, 0] = RuleBasic(Antecedent(child_antecedent_indices), None)
                else:
                    if random.random() < 0.5:
                        Y[0, k, 0] = a.copy()
                    else:
                        Y[0, k, 0] = b.copy()
            return Y

    class BasicMutation(Mutation):
        __learner = None

        def __init__(self, training_set):
            super().__init__()
            self.__learner = LearningBasic(training_set)

        def _do(self, problem, X, **kwargs):
            # for each individual
            for i in range(len(X)):
                indices = X[i, 0].get_antecedent().get_antecedent_indices()
                for j in range(len(indices)):
                    num_fuzzy_sets = Knowledge.get_instance().get_num_fuzzy_sets(j)
                    indices[j] = random.randint(0, num_fuzzy_sets - 1)

                X[i, 0].set_consequent(self.__learner.learning(X[i, 0].get_antecedent()))
            return X

    class BasicDuplicateElimination(ElementwiseDuplicateElimination):
        def is_equal(self, a, b):
            return False

    @staticmethod
    def hybrid_style_mofgbml(train, test):
        random.seed(2022)
        HomoTriangleKnowledgeFactory.create2_3_4_5(train.get_n_dim())

        bounds_Michigan = MichiganSolution.make_bounds()

        num_objectives_michigan = 1
        num_constraints_michigan = 0

        num_vars_pittsburgh = Consts.INITIATION_RULE_NUM
        num_objectives_pittsburgh = 2
        num_constraints_pittsburgh = 0

        rule_builder = RuleBasic.RuleBuilderBasic(HeuristicAntecedentFactory(train), LearningBasic(train))
        michigan_solution_builder = MichiganSolution.MichiganSolutionBuilder(bounds_Michigan,
                                                                             num_objectives_michigan,
                                                                             num_constraints_michigan,
                                                                             rule_builder)

        classification = SingleWinnerRuleSelection()
        classifier = Classifier(classification)

        problem = MoFGBMLBasicMain.BasicProblem(num_vars_pittsburgh,
                                                num_objectives_pittsburgh,
                                                num_constraints_pittsburgh,
                                                train,
                                                michigan_solution_builder,
                                                classifier)

        crossover_probability = 1

        algorithm = NSGA2(pop_size=Consts.POPULATION_SIZE,
                          sampling=MoFGBMLBasicMain.BasicSampling(train),
                          crossover=MoFGBMLBasicMain.BasicCrossover(2, 1, crossover_probability),
                          mutation=MoFGBMLBasicMain.BasicMutation(train),
                          eliminate_duplicates=MoFGBMLBasicMain.BasicDuplicateElimination())

        res = minimize(problem,
                       algorithm,
                       ('n_gen', 10),
                       seed=1,
                       verbose=True)

        cl = Classifier(SingleWinnerRuleSelection())
        res.X = [item[0] for item in res.X]
        non_dominated_solutions = res.pop

        print(res.X)
        for i in range(len(res.X)):
            print(res.X[i].get_class_label().get_class_label_value(), res.X[i].get_rule_weight().get_value())

        # print(cl.classify(res.X, train.get_patterns()[0]))


if __name__ == '__main__':
    MoFGBMLBasicMain.main(sys.argv[1:])
