//
// Created by Robin on 16/07/2025.
//

#ifndef RULE_BUILDER_BASIC_HPP
#define RULE_BUILDER_BASIC_HPP
#include "rule_builder_core.hpp"
#include "../rule_basic.hpp"


class RuleBuilderBasic : public RuleBuilderCore {
public:
    RuleBasic* create(Antecedent& antecedent) const override;
    RuleBuilderBasic* clone() const override;
};



#endif //RULE_BUILDER_BASIC_HPP
