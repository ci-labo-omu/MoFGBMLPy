from mofgbmlpy.fuzzy.rule.consequent.ruleWeight.abstract_rule_weight import AbstractRuleWeight


class RuleWeightBasic(AbstractRuleWeight):
    def __init__(self, rule_weight):
        self.set_value(rule_weight)

    def __copy__(self):
        return RuleWeightBasic(self.get_value())

    def __str__(self):
        if self.get_value() is None:
            raise ValueError("Rule weight is None")
        return f"{self.get_value():.4f}"
