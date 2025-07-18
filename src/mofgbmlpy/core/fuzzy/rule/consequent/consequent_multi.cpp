#include "consequent_multi.hpp"


ConsequentMulti::ConsequentMulti(std::shared_ptr<ClassLabelMulti> class_label,
                                 std::shared_ptr<RuleWeightMulti> rule_weight)
    : class_label(class_label), rule_weight(rule_weight) {}

ConsequentMulti::ConsequentMulti(const ConsequentMulti& other)
    : AbstractConsequent(other),
      class_label(std::make_shared<ClassLabelMulti>(*other.class_label)),
      rule_weight(std::make_shared<RuleWeightMulti>(*other.rule_weight)) {}

std::shared_ptr<AbstractClassLabel> ConsequentMulti::get_class_label() const {
    return class_label;
}

void ConsequentMulti::set_class_label_value(std::vector<int> value) {
    class_label->set_class_label_value(value);
}

std::vector<int> ConsequentMulti::get_class_label_value() const {
    return class_label->get_class_label_value();
}

std::shared_ptr<AbstractRuleWeight> ConsequentMulti::get_rule_weight() const {
    return rule_weight;
}

void ConsequentMulti::set_rule_weight(std::shared_ptr<AbstractRuleWeight> rw) {
    rule_weight = std::dynamic_pointer_cast<RuleWeightMulti>(rw);
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
