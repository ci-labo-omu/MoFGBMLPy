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
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.selection import Selection
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.termination import Termination
from pymoo.core.sampling import Sampling
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from src.fuzzy.knowledge.knowledge import Context
import random
import numpy as np

from src.fuzzy.rule.consequent.learning.learning_basic import LearningBasic
from src.fuzzy.rule.rule_basic import RuleBasic
from src.fuzzy.rule.antecedent.antecedent import Antecedent
from src.fuzzy.rule.antecedent.factory.all_combination_antecedent_factory import AllCombinationAntecedent

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

    class BasicProblem(ElementwiseProblem):
        def __init__(self):
            super().__init__(n_var=1, n_obj=1)

        def _evaluate(self, X, out, *args, **kwargs):
            out["F"] = np.array(len(X))
            for i in range(len(X)):
                out["F"][i] = X[i].evaluate()

    class BasicSampling(Sampling):
        def _do(self, problem, n_samples, **kwargs):
            X = np.full((n_samples, 1), None, dtype=object)
            factory = AllCombinationAntecedent()
            antecedents = factory.create(n_samples)
            for i in range(n_samples):
                X[i, 0] = RuleBasic(antecedents[i], LearningBasic.learning(antecedents[i]))

            return X

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
            # TODO: check https://pymoo.org/customization/custom.html
            Y = np.zeros((1, n_matings, n_var), dtype=object)

            # for each mating provided
            for k in range(n_matings):
                # get the first and the second parent
                a, b = X[0, k, 0], X[1, k, 0]
                if random.random() < self.prob:
                    p1_antecedent = X[0,].get_antecedent()
                    p2_antecedent = X[1].get_antecedent()

                    child_antecedent_indices = []
                    for i in range(X[0].get_antecedent().get_antecedent_length()):
                        if random.random() < 0.5:
                            child_antecedent_indices.append(p2_antecedent[i])
                        else:
                            child_antecedent_indices.append(p1_antecedent[i])
                    Y[0, k, 0] = RuleBasic(Antecedent(child_antecedent_indices), None)
                else:
                    if random.random() < 0.5:
                        Y[0, k, 0] = a.copy()
                    else:
                        Y[0, k, 0] = b.copy()
            return Y

    class BasicMutation(Mutation):
        def __init__(self):
            super().__init__()

        def _do(self, problem, X, **kwargs):
            # for each individual
            for i in range(len(X)):
                indices = X[i, 0].get_antecedent().get_antecedent_indices()
                for j in range(len(indices)):
                    num_fuzzy_sets = Context.get_instance().get_fuzzy_set_num(i)
                    if random.random() < self.prob:
                        indices[j] = random.randint(0, num_fuzzy_sets - 1)
            return X

    @staticmethod
    def hybrid_style_mofgbml(train, test):
        problem = MoFGBMLBasicMain.BasicProblem()
        algorithm = NSGA2(pop_size=Consts.POPULATION_SIZE, sampling=MoFGBMLBasicMain.BasicSampling(), crossover=MoFGBMLBasicMain.BasicCrossover(2, 1), mutation=MoFGBMLBasicMain.BasicMutation())

        res = minimize(problem,
                       algorithm,
                       ('n_gen', 200),
                       seed=1,
                       verbose=False)


if __name__ == '__main__':
    MoFGBMLBasicMain.main(sys.argv[1:])
