from fuzzy.classifier.classification.abstract_classification import AbstractClassification


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

    def get_error_rate(self, michigan_solution_list, dataset):
        num_errors = 0
        errored_patterns = []
        for pattern in dataset.get_patterns():
            winner_solution = self.classify(michigan_solution_list, pattern)

            if winner_solution is None:
                num_errors += 1
                errored_patterns.append(pattern)
                continue

            if pattern.get_target_class() != winner_solution.get_class_label():
                num_errors += 1
                errored_patterns.append(pattern)
        return num_errors / dataset.get_size(), errored_patterns

    @staticmethod
    def get_rule_num(michigan_solution_list):
        return len(michigan_solution_list)
