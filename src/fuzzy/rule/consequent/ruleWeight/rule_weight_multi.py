from src.fuzzy.rule.consequent.ruleWeight.abstract_rule_weight import AbstractRuleWeight


class RuleWeightMulti(AbstractRuleWeight):
    def __init__(self, rule_weight):
        self.set_rule_weight(rule_weight)

    def copy(self):
        return RuleWeightMulti(self.get_rule_weight())

    def __str__(self):
        if self.get_rule_weight() is None:
            raise ValueError("Rule weight is None")

        txt = f"{self.get_rule_weight()[0]:.4f}"

        if self.get_length() > 1:
            for i in range(1, self.get_length()):
                txt = f"{txt}, {f"{self.get_rule_weight()[i]:.4f}"}"

        return txt

    def get_rule_weight_at(self, index):
        return self.get_rule_weight()[index]

    def get_length(self):
        return len(self.get_rule_weight())