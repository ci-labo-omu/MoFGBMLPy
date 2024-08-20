import copy
from mofgbmlpy.data.class_label.abstract_class_label cimport AbstractClassLabel
from mofgbmlpy.data.class_label.class_label_multi import ClassLabelMulti
from mofgbmlpy.fuzzy.rule.consequent.ruleWeight.abstract_rule_weight cimport AbstractRuleWeight
from mofgbmlpy.data.class_label.class_label_multi cimport ClassLabelMulti
from mofgbmlpy.fuzzy.rule.consequent.ruleWeight.rule_weight_multi cimport RuleWeightMulti
from mofgbmlpy.fuzzy.rule.consequent.abstract_consequent cimport AbstractConsequent
cimport numpy as cnp

from mofgbmlpy.fuzzy.rule.consequent.ruleWeight.rule_weight_multi import RuleWeightMulti

cdef class ConsequentMulti(AbstractConsequent):
    cdef ClassLabelMulti _class_label
    cdef RuleWeightMulti _rule_weight

    cpdef AbstractClassLabel get_class_label(self)
    cpdef void set_class_label_value(self, object class_label_value)
    cpdef object get_class_label_value(self)
    cpdef AbstractRuleWeight get_rule_weight(self)
    cpdef void set_rule_weight(self, AbstractRuleWeight rule_weight)
    