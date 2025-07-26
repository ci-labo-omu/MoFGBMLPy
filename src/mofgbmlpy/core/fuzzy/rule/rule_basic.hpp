//
// Created by Robin on 16/07/2025.
//

#ifndef RULE_BASIC_HPP
#define RULE_BASIC_HPP
#include <vector>
#include "abstract_rule.hpp"


class RuleBasic : public AbstractRule {
public:
    RuleBasic(Antecedent* antecedent, AbstractConsequent* consequent);
    RuleBasic(const RuleBasic& other);
    double get_fitness_value(const std::vector<double>& attribute_vector) const override;
    operator std::string() const override;
    RuleBasic* clone() const override;
};



#endif //RULE_BASIC_HPP
