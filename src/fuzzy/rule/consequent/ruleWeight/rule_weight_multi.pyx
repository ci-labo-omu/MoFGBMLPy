from fuzzy.rule.consequent.ruleWeight.abstract_rule_weight import AbstractRuleWeight


class RuleWeightMulti(AbstractRuleWeight):
    def __init__(self, rule_weight):
        self.set_value(rule_weight)

    def __copy__(self):
        return RuleWeightMulti(self.get_value())

    def __str__(self):
        if self.get_value() is None:
            raise ValueError("Rule weight is None")

        txt = f"{self.get_value()[0]:.4f}"

        if self.get_length() > 1:
            for i in range(1, self.get_length()):
                txt = f"{txt}, {self.get_value()[i]:.4f}"

        return txt

    def get_rule_weight_at(self, index):
        return self.get_value()[index]

    def get_length(self):
        return len(self.get_value())