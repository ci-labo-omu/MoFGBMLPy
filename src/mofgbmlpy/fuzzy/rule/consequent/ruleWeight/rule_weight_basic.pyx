import xml.etree.cElementTree as xml_tree
from mofgbmlpy.fuzzy.rule.consequent.ruleWeight.abstract_rule_weight cimport AbstractRuleWeight


cdef class RuleWeightBasic(AbstractRuleWeight):
    def __init__(self, double rule_weight):
        self.__rule_weight = rule_weight

    def __deepcopy__(self, memo={}):
        new_object = RuleWeightBasic(self.get_value())
        memo[id(self)] = new_object
        return new_object

    def __str__(self):
        if self.get_value() is None:
            # with cython.gil:
            raise ValueError("Rule weight is None")
        return f"{self.get_value():.4f}"

    cpdef object get_value(self):
        return self.__rule_weight

    cpdef void set_value(self, object rule_weight):
        self.__rule_weight = rule_weight

    def __eq__(self, other):
        return self.__rule_weight == other.get_value()

    def to_xml(self):
        root = xml_tree.Element("ruleWeight")
        root.text = str(self)

        return root