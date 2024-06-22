from abc import ABC


class AbstractRuleWeight(ABC):
    _rule_weight = None

    def get_value(self):
        return self._rule_weight

    def set_value(self, rule_weight):
        self._rule_weight = rule_weight
