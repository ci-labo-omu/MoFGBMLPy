import copy

from mofgbmlpy.fuzzy.classifier.classification.single_winner_rule_selection import SingleWinnerRuleSelection


def test_deep_copy():
    # Just check if it raises an exception
    obj = SingleWinnerRuleSelection()
    _ = copy.deepcopy(obj)

    assert True
