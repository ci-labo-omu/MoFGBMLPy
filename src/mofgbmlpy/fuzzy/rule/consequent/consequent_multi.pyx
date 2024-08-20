import xml.etree.cElementTree as xml_tree
import copy

from mofgbmlpy.data.class_label.abstract_class_label cimport AbstractClassLabel
from mofgbmlpy.fuzzy.rule.consequent.ruleWeight.abstract_rule_weight cimport AbstractRuleWeight
from mofgbmlpy.data.class_label.class_label_multi cimport ClassLabelMulti
from mofgbmlpy.fuzzy.rule.consequent.ruleWeight.rule_weight_multi cimport RuleWeightMulti
from mofgbmlpy.fuzzy.rule.consequent.abstract_consequent cimport AbstractConsequent

cdef class ConsequentMulti(AbstractConsequent):
    """Consequent part of a fuzzy rule for multilabel classification

    Attributes:
        _class_label (ClassLabelMulti): Class label of this consequent
        _rule_weight (RuleWeightMulti): Rule weight of this consequent
    """
    def __init__(self, ClassLabelMulti class_label, RuleWeightMulti rule_weight):
        """Constructor

        Args:
            class_label (ClassLabelMulti): Class label of this consequent
            rule_weight (RuleWeightMulti): Rule weight of this consequent
        """
        self._class_label = class_label
        self._rule_weight = rule_weight
        super().__init__()

    cpdef AbstractClassLabel get_class_label(self):
        """Get the class label

        Returns:
            AbstractClassLabel: Class label
        """
        return self._class_label


    cpdef AbstractRuleWeight get_rule_weight(self):
        """Get the rule weight object

        Returns:
            AbstractRuleWeight: Rule weight object
        """
        return self._rule_weight

    cpdef void set_rule_weight(self, AbstractRuleWeight rule_weight):
        """Set the rule weight object

        Args:
            rule_weight (AbstractRuleWeight): New rule weight object
        """
        self._rule_weight = rule_weight

    def __deepcopy__(self, memo={}):
        """Return a deepcopy of this object

        Args:
            memo (dict): Dictionary of objects already copied during the current copying pass;

        Returns:
            object: Deep copy of this object
        """
        new_consequent = ConsequentMulti(copy.deepcopy(self._class_label), copy.deepcopy(self._rule_weight))
        memo[id(self)] = new_consequent
        return new_consequent

    def __eq__(self, other):
        """Check if another object is equal to this one

        Args:
            other (object): Object compared to this one

        Returns:
            (bool) True if they are equal and False otherwise
        """
        if not isinstance(other, ConsequentMulti):
            return False

        return self._class_label == other.get_class_label() and self._rule_weight == other.get_rule_weight()
