import copy

from pymoo.core.callback import Callback

from mofgbmlpy.fuzzy.classification.single_winner_rule_selection import SingleWinnerRuleSelection
from mofgbmlpy.gbml.solution.pittsburgh_solution import PittsburghSolution


class RestorePopulationIfWorseMichigan(Callback):
    def __init__(self, training_set):
        super().__init__()
        self._best_pop = None
        self._best_err_rate = None
        self._training_set = training_set

    def notify(self, algorithm):
        current_pop = algorithm.pop
        michigan_solutions = current_pop.get("X")[:, 0]

        classifier = PittsburghSolution(num_vars=len(current_pop),
                                        num_objectives=0,
                                        num_constraints=0,
                                        classification=SingleWinnerRuleSelection(),
                                        do_init_vars=False)

        classifier.set_vars(michigan_solutions)
        current_err_rate = classifier.get_error_rate(self._training_set)

        if self._best_err_rate is None or self._best_err_rate > current_err_rate:
            self._best_pop = copy.deepcopy(current_pop)  # TODO: Might not be a deep copy here
            self._best_err_rate = current_err_rate
        else:
            algorithm.pop = copy.deepcopy(self._best_pop)
