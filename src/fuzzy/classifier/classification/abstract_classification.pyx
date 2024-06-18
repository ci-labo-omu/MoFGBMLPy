from abc import abstractmethod, ABC


class AbstractClassification(ABC):
    @abstractmethod
    def classify(self, michigan_solution_list, pattern):
        pass

    @abstractmethod
    def __copy__(self):
        pass
