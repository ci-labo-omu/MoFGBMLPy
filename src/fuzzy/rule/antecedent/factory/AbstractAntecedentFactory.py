from abc import abstractmethod, ABC


class AbstractAntecedentFactory(ABC):
    @abstractmethod
    def create(self, num_rules=1):
        pass

    @abstractmethod
    def copy(self):
        pass
