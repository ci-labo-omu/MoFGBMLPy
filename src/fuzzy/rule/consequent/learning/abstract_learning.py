from abc import ABC, abstractmethod


class AbstractLearning(ABC):
    _default_reject_threshold = 0

    @abstractmethod
    def learning(self, antecedent, reject_threshold=_default_reject_threshold):
        pass

    @abstractmethod
    def copy(self):
        pass
