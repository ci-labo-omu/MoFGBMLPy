import copy


class Consequent:
    _class_label = None
    _rule_weight = None

    def __init__(self, class_label, rule_weight):
        self._class_label = class_label
        self._rule_weight = rule_weight

    def get_class_label(self):
        return self._class_label

    def set_class_label_value(self, class_label_value):
        self._class_label.set_class_label_value(class_label_value)

    def get_class_label_value(self):
        return self._class_label.get_class_label_value()

    def __eq__(self, other):
        return self.get_class_label_value() == other.get_class_label_value()

    def is_rejected(self):
        return self.get_class_label().is_rejected()

    def set_rejected(self):
        self._class_label.set_rejected()

    def get_rule_weight(self):
        return self._rule_weight

    def set_rule_weight(self, rule_weight):
        self._rule_weight = rule_weight

    def __copy__(self):
        return Consequent(copy.copy(self._class_label), copy.copy(self._rule_weight))

    def __str__(self):
        return f"class:[{self._class_label}]: weight:[{self._rule_weight}]"
