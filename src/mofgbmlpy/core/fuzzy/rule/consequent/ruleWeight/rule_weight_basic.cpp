#include "rule_weight_basic.hpp"

#include <string>


RuleWeightBasic::RuleWeightBasic(double rule_weight)
    : rule_weight(rule_weight) {}

RuleWeightBasic::RuleWeightBasic(const RuleWeightBasic& other): AbstractRuleWeight(other), rule_weight(other.rule_weight) {}

std::variant<double, std::vector<double>> RuleWeightBasic::get_value() const {
    return rule_weight;
}

void RuleWeightBasic::set_value(std::variant<double, std::vector<double>> new_rule_weight) {
    this->rule_weight = std::get<double>(new_rule_weight);
}

double RuleWeightBasic::get_raw_value() const {
    return rule_weight;
}

RuleWeightBasic::operator std::string() const
{
    return std::to_string(rule_weight);
}

bool RuleWeightBasic::operator==(const AbstractRuleWeight& other) const
{
    if (auto other_basic = dynamic_cast<const RuleWeightBasic*>(&other)) {
        return rule_weight == other_basic->rule_weight;
    }
    return false;
}

RuleWeightBasic* RuleWeightBasic::clone() const
{
    return new RuleWeightBasic(*this);
}

