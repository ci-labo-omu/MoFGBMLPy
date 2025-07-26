#include "consequent_basic.hpp"

ConsequentBasic::ConsequentBasic(ClassLabelBasic* class_label,
                                 RuleWeightBasic* rule_weight)
    : class_label(class_label), rule_weight(rule_weight) {}

ConsequentBasic::ConsequentBasic(const ConsequentBasic& other)
    : AbstractConsequent(other),
      class_label(new ClassLabelBasic(*other.class_label)),
      rule_weight(new RuleWeightBasic(*other.rule_weight)) {}

AbstractClassLabel* ConsequentBasic::get_class_label() const {
    return class_label;
}

void ConsequentBasic::set_class_label_value(int value) {
    class_label->set_class_label_value(value);
}

int ConsequentBasic::get_class_label_value() const {
    return class_label->get_class_label_value();
}

AbstractRuleWeight* ConsequentBasic::get_rule_weight() const {
    return rule_weight;
}

void ConsequentBasic::set_rule_weight(AbstractRuleWeight* rw) {
    rule_weight = dynamic_cast<RuleWeightBasic*>(rw);
}

ConsequentBasic* ConsequentBasic::clone() const
{
    return new ConsequentBasic(*this);
}

bool ConsequentBasic::operator==(const AbstractConsequent& other) const
{
    if (auto other_basic = dynamic_cast<const ConsequentBasic*>(&other)) {
        return *class_label == *other_basic->class_label && *rule_weight == *other_basic->rule_weight;
    }
    return false;
}

