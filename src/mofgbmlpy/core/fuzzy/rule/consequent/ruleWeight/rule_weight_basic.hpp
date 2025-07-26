#ifndef RULE_WEIGHT_BASIC_HPP
#define RULE_WEIGHT_BASIC_HPP

#include <memory>
#include "abstract_rule_weight.hpp"


class RuleWeightBasic : public AbstractRuleWeight {
private:
    double rule_weight;

public:
    RuleWeightBasic(double rule_weight);
    RuleWeightBasic(const RuleWeightBasic& other);
    std::variant<double, std::vector<double>> get_value() const override;
    void set_value(std::variant<double, std::vector<double>> new_rule_weight) override;
    double get_raw_value() const;

    operator std::string() const override;
    bool operator==(const AbstractRuleWeight& other) const override;
    RuleWeightBasic* clone() const override;
};


#endif  // RULE_WEIGHT_BASIC_HPP
