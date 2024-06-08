from src.fuzzy.classifier.classification.abstract_classification import AbstractClassification


class Classifier:
    _classification = None

    def __init__(self, classification):
        if classification is None or not isinstance(classification, AbstractClassification):
            raise Exception("Invalid classification method")
        self._classification = classification

    def classify(self, michigan_solution_list, pattern):
        return self._classification.classify(michigan_solution_list, pattern)

    def __copy__(self):
        return Classifier(self._classification)

    @staticmethod
    def get_rule_length(michigan_solution_list):
        length = 0
        for item in michigan_solution_list:
            length += item.get_rule_length()
        return length

    @staticmethod
    def get_rule_num(michigan_solution_list):
        return len(michigan_solution_list)
