#include "rule_weight_multi.hpp"

#include <string>


RuleWeightMulti::RuleWeightMulti(const std::vector<double>& rule_weight)
    : rule_weight(rule_weight) {}

RuleWeightMulti::RuleWeightMulti(const RuleWeightMulti& other): AbstractRuleWeight(other), rule_weight(other.rule_weight) {}

std::variant<double, std::vector<double>> RuleWeightMulti::get_value() const {
    return rule_weight;
}

void RuleWeightMulti::set_value(std::variant<double, std::vector<double>> value) {
    rule_weight = std::get<std::vector<double>>(value);
}

double RuleWeightMulti::get_rule_weight_at(int index) const {
    return rule_weight.at(index);
}

int RuleWeightMulti::get_length() const {
    return rule_weight.size();
}

double RuleWeightMulti::get_mean() const {
    double sum = 0.0;
    for (double val : rule_weight) {
        sum += val;
    }
    return rule_weight.empty() ? 0.0 : sum / rule_weight.size();
}

RuleWeightMulti::operator std::string() const
{
    std::string txt = std::to_string(rule_weight[0]);
    if (rule_weight.size() > 1) {
        for (size_t i = 1; i < rule_weight.size(); ++i) {
            txt += ", " + std::to_string(rule_weight[i]);
        }
    }
    return txt;
}

bool RuleWeightMulti::operator==(const AbstractRuleWeight& other) const
{
    const RuleWeightMulti* other_multi = dynamic_cast<const RuleWeightMulti*>(&other);
    if (!other_multi) {
        return false;
    }
    return rule_weight == other_multi->rule_weight;
}

RuleWeightMulti* RuleWeightMulti::clone() const
{
    return new RuleWeightMulti(*this);
}
