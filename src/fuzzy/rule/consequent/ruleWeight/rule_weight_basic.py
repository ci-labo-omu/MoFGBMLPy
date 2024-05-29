from src.fuzzy.rule.consequent.ruleWeight.abstract_rule_weight import AbstractRuleWeight


class RuleWeightBasic(AbstractRuleWeight):
    def __init__(self, rule_weight):
        self.set_rule_weight(rule_weight)

    def copy(self):
        return RuleWeightBasic(self.get_rule_weight())

    def __str__(self):
        if self.get_rule_weight() is None:
            raise ValueError("Rule weight is None")
        return f"{self.get_rule_weight():.4f}"
