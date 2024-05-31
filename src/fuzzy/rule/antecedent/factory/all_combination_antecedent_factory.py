from src.fuzzy.rule.antecedent.factory.abstract_antecedent_factory import AbstractAntecedentFactory
from src.fuzzy.knowledge.knowledge import Context
from src.fuzzy.rule.antecedent.antecedent import Antecedent
import numpy as np


class AllCombinationAntecedent(AbstractAntecedentFactory):
    __antecedents = None
    __dimension = None

    def __init__(self):
        self.__dimension = Context.get_instance().get_num_dim()
        self.generate_antecedents(Context.get_instance().get_fuzzy_sets())

    def generate_antecedents(self, fuzzy_sets):
        queue = [[]]
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

        self.__antecedents = np.zeros((len(indices), self.__dimension), dtype=int)
        for i in range(len(indices)):
            self.__antecedents[i, :] = np.array(indices[i], dtype=int)

    def create(self, num_rules=1):
        num_rules = min(num_rules, len(self.__antecedents))
        # Return an antecedent
        if self.__antecedents is None:
            raise Exception("AllCombinationAntecedentFactory hasn't been initialised")
        indices = np.random.choice(list(range(len(self.__antecedents))), num_rules, replace=False)
        antecedent_objects = np.zeros((num_rules), dtype=object)

        for i in range(num_rules):
            antecedent_objects[i] = Antecedent(np.copy(self.__antecedents[indices[i], :]))

        if num_rules == 1:
            antecedent_objects = antecedent_objects[0]
        return antecedent_objects

    def __str__(self):
        return "AllCombinationAntecedentFactory [antecedents=" + str(self.__antecedents) + ", dimension=" + str(self.__dimension) + "]"

    def copy(self):
        return AllCombinationAntecedent()
