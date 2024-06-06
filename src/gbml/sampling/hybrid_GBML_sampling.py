from pymoo.core.sampling import Sampling


class HybridGBMLSampling(Sampling):
    __learner = None

    def __init__(self, training_set):
        super().__init__()
        self.__learner = LearningBasic(training_set)

    def _do(self, problem, n_samples, **kwargs):
        return np.array([problem.create_solution()] * n_samples, dtype=object)