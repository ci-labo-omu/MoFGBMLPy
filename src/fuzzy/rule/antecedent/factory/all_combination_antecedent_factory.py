import copy

from src.fuzzy.rule.antecedent.factory.abstract_antecedent_factory import AbstractAntecedentFactory
from src.fuzzy.knowledge.knowledge import Knowledge
from src.fuzzy.rule.antecedent.antecedent import Antecedent
import numpy as np


class AllCombinationAntecedentFactory(AbstractAntecedentFactory):
    __antecedents_indices = None
    __dimension = None

    def __init__(self):
        self.__dimension = Knowledge.get_instance().get_num_dim()
        self.generate_antecedents_indices(Knowledge.get_instance().get_fuzzy_sets())

    def generate_antecedents_indices(self, fuzzy_sets):
        queue = [[]]
        indices = []

        # Generate all combination of fuzzy sets indices
        while len(queue) > 0:
            buffer = queue.pop(0)
            current_dim = len(buffer)
            if current_dim < self.__dimension:
                for i in range(len(fuzzy_sets[current_dim])):
                    tmp = copy.copy(buffer)
                    tmp.append(i)
                    queue.append(tmp)
            else:
                indices.append(buffer)

        self.__antecedents_indices = np.zeros((len(indices), self.__dimension), dtype=np.int_)
        for i in range(len(indices)):
            self.__antecedents_indices[i, :] = np.array(indices[i], dtype=np.int_)

    def create(self, num_rules=1):
        antecedent_objects = np.zeros(num_rules, dtype=object)
        indices = self.create_antecedent_indices(num_rules)

        for i in range(num_rules):
            antecedent_objects[i] = Antecedent(np.copy(self.__antecedents_indices[indices[i], :]))

        if num_rules == 1:
            antecedent_objects = antecedent_objects[0]
        return antecedent_objects

    def create_antecedent_indices(self, num_rules=1):
        num_rules = min(num_rules, len(self.__antecedents_indices))
        # Return an antecedent
        if self.__antecedents_indices is None:
            raise Exception("AllCombinationAntecedentFactory hasn't been initialised")
        return np.random.choice(list(range(len(self.__antecedents_indices))), num_rules, replace=False)

    def __str__(self):
        return "AllCombinationAntecedentFactory [antecedents=" + str(self.__antecedents_indices) + ", dimension=" + str(
            self.__dimension) + "]"

    def __copy__(self):
        return AllCombinationAntecedentFactory()
