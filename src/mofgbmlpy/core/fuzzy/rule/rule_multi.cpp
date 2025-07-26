//
// Created by Robin on 16/07/2025.
//

#include "rule_multi.hpp"

RuleMulti::RuleMulti(Antecedent* antecedent, AbstractConsequent* consequent): AbstractRule(std::move(antecedent), std::move(consequent)) {}

RuleMulti::RuleMulti(const RuleMulti& other): AbstractRule(other) {}

double RuleMulti::get_fitness_value(const std::vector<double>& attribute_vector) const
{
    double membership_value = antecedent->get_compatible_grade_value(attribute_vector);
    RuleWeightMulti* rule_weight_multi = dynamic_cast<RuleWeightMulti*>(get_rule_weight());
    return membership_value * rule_weight_multi->get_mean();
}

RuleMulti::operator std::string() const
{
    return "Rule_MultiClass [antecedent=" + std::string(*antecedent) +
           ", consequent=" + std::string(*consequent) + "]";
}

RuleMulti* RuleMulti::clone() const
{
    return new RuleMulti(*this);
}