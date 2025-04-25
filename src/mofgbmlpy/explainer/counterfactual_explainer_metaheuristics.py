from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from mofgbmlpy.explainer.gbml.problem.counterfactual_problem import CounterfactualProblem
from mofgbmlpy.explainer.gbml.fuzzy_sets_sampling import FuzzySetsSampling
from mofgbmlpy.explainer.gbml.operators.fuzzy_sets_repair import FuzzySetsRepair
from mofgbmlpy.explainer.gbml.operators.fuzzy_sets_mutation import FuzzySetsMutation
from mofgbmlpy.explainer.gbml.operators.fuzzy_sets_crossover import FuzzySetsCrossover
from mofgbmlpy.fuzzy.rule.consequent.learning.learning_basic import LearningBasic
from mofgbmlpy.fuzzy.knowledge.factory.homo_triangle_knowledge_factory_2_3_4_5 import (
    HomoTriangleKnowledgeFactory_2_3_4_5,
)
from mofgbmlpy.main.nsgaii.mofgbml_nsgaii_main import MoFGBMLNSGAIIMain
from mofgbmlpy.data.class_label.class_label_basic import ClassLabelBasic


class CounterFactualExplainerMetaheuristics:
    def __init__(self, fuzzy_rule, target_class, learner):
        initial_knowledge = fuzzy_rule.get_antecedent().get_knowledge()
        initial_class = fuzzy_rule.get_class_label()

        self._problem = CounterfactualProblem(initial_knowledge, fuzzy_rule, initial_class, target_class, learner)
        self._repair = FuzzySetsRepair()
        self._sampling = FuzzySetsSampling()
        self._mutation = FuzzySetsMutation(0.5)
        self._crossover = FuzzySetsCrossover(0.5)

    def train(self):
        pop_size = 30
        n_max_iters = 50

        algorithm = NSGA2(
            repair=self._repair,  # fix invalid params if any remaining
            pop_size=pop_size,
            sampling=self._sampling,
            crossover=self._crossover,
            mutation=self._mutation,  # should consider bounds and conditions of membership functions params
            eliminate_duplicates=True,
            n_max_iters=n_max_iters,
        )

        res = minimize(self._problem, algorithm, seed=41, verbose=True)

        return res


if __name__ == "__main__":
    args = [
        "--data-name",
        "appendicitis",
        "--algorithm-id",
        "0",
        "--experiment-id",
        "0",
        "--train-file",
        "..\\..\\..\\dataset\\appendicitis\\a0_0_appendicitis-10tra.dat",
        "--test-file",
        "..\\..\\..\\dataset\\appendicitis\\a0_0_appendicitis-10tra.dat",
        "--terminate-evaluation",
        "1000",
        "--no-output-files",
        "--objectives",
        "error-rate",
        "num-rules",
    ]

    runner = MoFGBMLNSGAIIMain(HomoTriangleKnowledgeFactory_2_3_4_5)
    res = runner.main(args)

    non_dominated_solutions = res.X
    objectives_non_dominated_solutions = res.F

    sol1 = non_dominated_solutions[0]
    rule = sol1[0].get_var(0).get_rule()

    learner = LearningBasic(runner.get_train_set())
    target_class = ClassLabelBasic(1)

    explainer = CounterFactualExplainerMetaheuristics(rule, target_class, learner)
    explainer.train()
