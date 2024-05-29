from abc import ABC


class AbstractRuleWeight(ABC):
    _rule_weight = None

    def get_rule_weight(self):
        return self._rule_weight

    def set_rule_weight(self, rule_weight):
        self._rule_weight = rule_weight
