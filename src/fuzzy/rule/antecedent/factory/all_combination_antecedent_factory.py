from src.fuzzy.rule.antecedent.factory.abstract_antecedent_factory import AbstractAntecedentFactory
from src.fuzzy.knowledge.knowledge import Context
from src.fuzzy.rule.antecedent.antecedent import Antecedent
import numpy as np


class AllCombinationAntecedent(AbstractAntecedentFactory):
    __antecedents = None
    __dimension = None

    def __init__(self):
        self.generate_antecedents(Context.get_instance().get_fuzzy_sets())
        self.__dimension = Context.get_instance().get_num_dim()

    def generate_antecedents(self, fuzzy_sets):
        queue = []
        indices = []

        # Generate all combination of fuzzy sets indices
        while len(queue) > 0:
            buffer = queue.pop(0)
            current_dim = len(buffer)
            if current_dim < self.__dimension:
                for i in range(len(fuzzy_sets[current_dim])):
                    tmp = buffer.copy()
                    tmp.append(i)
                    queue.append(tmp)
            else:
                indices.append(buffer)

        self.__antecedents = np.array((len(indices), self.__dimension))
        for i in range(len(indices)):
            self.__antecedents[i] = np.array(indices[i])

    def create(self, num_rules=1):
        # Return an antecedent
        if self.__antecedents is None:
            raise Exception("AllCombinationAntecedentFactory hasn't been initialised")
        antecedents_indices = np.random.choice(self.__antecedents, num_rules, replace=False)

        antecedents = np.array([Antecedent(np.copy(indices)) for indices in antecedents_indices], dtype=object)
        return antecedents

    def __str__(self):
        return "AllCombinationAntecedentFactory [antecedents=" + str(self.__antecedents) + ", dimension=" + str(self.__dimension) + "]"

    def copy(self):
        return AllCombinationAntecedent()
