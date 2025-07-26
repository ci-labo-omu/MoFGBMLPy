#include "consequent_multi.hpp"


ConsequentMulti::ConsequentMulti(ClassLabelMulti* class_label,
                                 RuleWeightMulti* rule_weight)
    : class_label(class_label), rule_weight(rule_weight) {}

ConsequentMulti::ConsequentMulti(const ConsequentMulti& other)
    : AbstractConsequent(other),
      class_label(new ClassLabelMulti(*other.class_label)),
      rule_weight(new RuleWeightMulti(*other.rule_weight)) {}

AbstractClassLabel* ConsequentMulti::get_class_label() const {
    return class_label;
}

void ConsequentMulti::set_class_label_value(std::vector<int> value) {
    class_label->set_class_label_value(value);
}

std::vector<int> ConsequentMulti::get_class_label_value() const {
    return class_label->get_class_label_value();
}

AbstractRuleWeight* ConsequentMulti::get_rule_weight() const {
    return rule_weight;
}

void ConsequentMulti::set_rule_weight(AbstractRuleWeight* rw) {
    rule_weight = dynamic_cast<RuleWeightMulti*>(rw);
}

ConsequentMulti* ConsequentMulti::clone() const
{
    return new ConsequentMulti(*this);
}

bool ConsequentMulti::operator==(const AbstractConsequent& other) const
{
    if (auto other_multi = dynamic_cast<const ConsequentMulti*>(&other)) {
        return *class_label == *other_multi->class_label && *rule_weight == *other_multi->rule_weight;
    }
    return false;
}
