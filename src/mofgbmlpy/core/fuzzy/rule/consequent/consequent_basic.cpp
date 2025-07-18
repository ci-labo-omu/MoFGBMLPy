#include "consequent_basic.hpp"

ConsequentBasic::ConsequentBasic(std::shared_ptr<ClassLabelBasic> class_label,
                                 std::shared_ptr<RuleWeightBasic> rule_weight)
    : class_label(class_label), rule_weight(rule_weight) {}

ConsequentBasic::ConsequentBasic(const ConsequentBasic& other)
    : AbstractConsequent(other),
      class_label(std::make_shared<ClassLabelBasic>(*other.class_label)),
      rule_weight(std::make_shared<RuleWeightBasic>(*other.rule_weight)) {}

std::shared_ptr<AbstractClassLabel> ConsequentBasic::get_class_label() const {
    return class_label;
}

void ConsequentBasic::set_class_label_value(int value) {
    class_label->set_class_label_value(value);
}

int ConsequentBasic::get_class_label_value() const {
    return class_label->get_class_label_value();
}

std::shared_ptr<AbstractRuleWeight> ConsequentBasic::get_rule_weight() const {
    return rule_weight;
}

void ConsequentBasic::set_rule_weight(std::shared_ptr<AbstractRuleWeight> rw) {
    rule_weight = std::dynamic_pointer_cast<RuleWeightBasic>(rw);
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

