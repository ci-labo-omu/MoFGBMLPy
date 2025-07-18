//
// Created by Robin on 16/07/2025.
//

#include "rule_basic.hpp"

RuleBasic::RuleBasic(std::shared_ptr<Antecedent> antecedent, std::shared_ptr<AbstractConsequent> consequent): AbstractRule(std::move(antecedent), std::move(consequent)) {}

RuleBasic::RuleBasic(const RuleBasic& other): AbstractRule(other) {}

double RuleBasic::get_fitness_value(const std::vector<double>& attribute_vector) const
{
    double membership_value = antecedent->get_compatible_grade_value(attribute_vector);
    double cf = std::get<double>(get_rule_weight()->get_value());
    return membership_value * cf;
}

RuleBasic::operator std::string() const
{
    return "Rule_Basic [antecedent=" + std::string(*antecedent) +
           ", consequent=" + std::string(*consequent) + "]";
}

RuleBasic* RuleBasic::clone() const
{
    return new RuleBasic(*this);
}
