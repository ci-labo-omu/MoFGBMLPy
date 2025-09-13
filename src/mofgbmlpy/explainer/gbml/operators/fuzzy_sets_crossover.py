import copy

from pymoo.core.crossover import Crossover
import numpy as np

from mofgbmlpy.explainer.gbml.problem.counterfactual_problem import CounterfactualProblem
from mofgbmlpy.fuzzy.fuzzy_term.fuzzy_set.dont_care_fuzzy_set import DontCareFuzzySet

from mofgbmlpy.fuzzy.fuzzy_term.fuzzy_set.triangular_fuzzy_set import TriangularFuzzySet

from mofgbmlpy.gbml.operator.crossover.pymoo_deepcopy_crossover import PymooDeepcopyCrossover


class FuzzySetsCrossover(PymooDeepcopyCrossover):
    def __init__(self, prob=0.5, p1_prob_off_1=0.5):
        super().__init__(n_parents=2, n_offsprings=1, prob=prob)
        self._p1_prob_off_1 = p1_prob_off_1

    def _do(self, problem, X, **kwargs):
        _, n_matings, num_vars = X.shape
        offsprings = np.zeros((1, n_matings, num_vars), dtype=object)

        for i in range(n_matings):
            p1 = X[0, i, :]
            p2 = X[1, i, :]
            offsprings[0, i, 0] = copy.deepcopy(p1[0])


            p1_fuzzy_sets = CounterfactualProblem.get_fuzzy_sets_from_rule(p1[0].get_rule())
            p2_fuzzy_sets = CounterfactualProblem.get_fuzzy_sets_from_rule(p2[0].get_rule())
            num_dims = p1[0].get_rule().get_antecedent_array_size()
            child_fuzzy_sets = np.empty(num_dims, dtype=object)
            child_antecedent_indices = np.zeros(num_dims, dtype=int)

            for j in range(num_dims):
                if np.random.rand() < self._p1_prob_off_1:
                    child_fuzzy_sets[j] = copy.deepcopy(p1_fuzzy_sets[j])
                else:
                    child_fuzzy_sets[j] = copy.deepcopy(p2_fuzzy_sets[j])

                if isinstance(child_fuzzy_sets[j], TriangularFuzzySet):
                    child_antecedent_indices[j] = 1
                elif not isinstance(child_fuzzy_sets[j], DontCareFuzzySet):
                    raise ValueError("Fuzzy set type not supported in crossover.")

            offsprings[0, i, 0].set_knowledge(CounterfactualProblem.build_knowledge(child_fuzzy_sets))
            offsprings[0, i, 0].set_vars(child_antecedent_indices)
            offsprings[0, i, 0].get_rule().get_antecedent().set_antecedent_indices(child_antecedent_indices)


            p1_knowledge = p1[0].get_rule().get_knowledge()
            p2_knowledge = p2[0].get_rule().get_knowledge()
            child_knowledge = offsprings[0, i, 0].get_rule().get_knowledge()

            if id(child_knowledge) == id(p1_knowledge) or id(child_knowledge) == id(p2_knowledge):
                raise ValueError("Knowledge object not copied properly in crossover.")
            for child_fvar in child_knowledge.get_fuzzy_vars():
                for child_fs in child_fvar.get_fuzzy_sets():
                    for p1_fvar in p1_knowledge.get_fuzzy_vars():
                        if id(child_fvar) == id(p1_fvar):
                            raise ValueError("Fuzzy variable object not copied properly in crossover.")
                        for p1_fs in p1_fvar.get_fuzzy_sets():
                            if id(child_fs) == id(p1_fs):
                                raise ValueError("Fuzzy set object not copied properly in crossover.")
                    for p2_fvar in p2_knowledge.get_fuzzy_vars():
                        if id(child_fvar) == id(p2_fvar):
                            raise ValueError("Fuzzy variable object not copied properly in crossover.")
                        for p2_fs in p2_fvar.get_fuzzy_sets():
                            if id(child_fs) == id(p2_fs):
                                raise ValueError("Fuzzy set object not copied properly in crossover.")


        return offsprings
