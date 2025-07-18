//
// Created by Robin on 16/07/2025.
//

#ifndef RULE_MULTI_HPP
#define RULE_MULTI_HPP
#include <memory>

#include "rule_basic.hpp"
#include "antecedent/antecedent.hpp"


class RuleMulti : public AbstractRule {
public:
    RuleMulti(std::shared_ptr<Antecedent> antecedent, std::shared_ptr<AbstractConsequent> consequent);
    RuleMulti(const RuleMulti& other);
    double get_fitness_value(const std::vector<double>& attribute_vector) const override;
    operator std::string() const override;
    RuleMulti* clone() const override;
};



#endif //RULE_MULTI_HPP
