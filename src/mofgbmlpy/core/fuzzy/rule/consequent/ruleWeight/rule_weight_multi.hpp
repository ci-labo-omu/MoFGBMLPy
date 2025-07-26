#ifndef RULE_WEIGHT_MULTI_HPP
#define RULE_WEIGHT_MULTI_HPP

#include <memory>
#include <vector>
#include "abstract_rule_weight.hpp"


class RuleWeightMulti : public AbstractRuleWeight {
private:
    std::vector<double> rule_weight;

public:
    RuleWeightMulti(const std::vector<double>& rule_weight);
    RuleWeightMulti(const RuleWeightMulti& other);
    std::variant<double, std::vector<double>> get_value() const override;
    void set_value(std::variant<double, std::vector<double>> rule_weight) override;

    double get_rule_weight_at(int index) const;
    int get_length() const;
    double get_mean() const;

    operator std::string() const override;
    bool operator==(const AbstractRuleWeight& other) const override;
    RuleWeightMulti* clone() const override;
};


#endif  // RULE_WEIGHT_MULTI_HPP
