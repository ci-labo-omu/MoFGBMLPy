import copy

from mofgbmlpy.fuzzy.classifier.classification.single_winner_rule_selection import SingleWinnerRuleSelection
from mofgbmlpy.fuzzy.classifier.classifier import Classifier


def test_deep_copy():
    # Just check if it raises an exception
    obj = Classifier(SingleWinnerRuleSelection())
    _ = copy.deepcopy(obj)

    assert True
