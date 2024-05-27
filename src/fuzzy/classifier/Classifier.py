from src.fuzzy.classifier.classification.AbstractClassification import AbstractClassification


class Classifier:
    _classification = None

    def __init__(self, classification):
        if classification is None or isinstance(classification, AbstractClassification):
            raise Exception("Invalid classification method")
        self._classification = classification

    def classify(self, michigan_solution_list, pattern):
        return self._classification.classify(michigan_solution_list, pattern)

    def copy(self):
        new_object = Classifier(self._classification)

    @staticmethod
    def get_rule_length(michigan_solution_list):
        length = 0
        for item in michigan_solution_list:
            length += item.get_rule_length()
        return length

    @staticmethod
    def get_rule_num(michigan_solution_list):
        return len(michigan_solution_list)
