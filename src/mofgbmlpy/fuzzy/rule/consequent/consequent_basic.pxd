import copy
from mofgbmlpy.data.class_label.abstract_class_label cimport AbstractClassLabel
from mofgbmlpy.data.class_label.class_label_basic cimport ClassLabelBasic
from mofgbmlpy.fuzzy.rule.consequent.abstract_consequent cimport AbstractConsequent
from mofgbmlpy.fuzzy.rule.consequent.ruleWeight.abstract_rule_weight cimport AbstractRuleWeight
cimport numpy as cnp

from mofgbmlpy.fuzzy.rule.consequent.ruleWeight.rule_weight_basic cimport RuleWeightBasic

cdef class ConsequentBasic(AbstractConsequent):
    cdef ClassLabelBasic _class_label
    cdef RuleWeightBasic _rule_weight

    cpdef AbstractClassLabel get_class_label(self)
    cpdef AbstractRuleWeight get_rule_weight(self)
    cpdef void set_rule_weight(self, AbstractRuleWeight rule_weight)
