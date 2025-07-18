//
// Created by Robin on 16/07/2025.
//

#ifndef RULE_BUILDER_MULTI_HPP
#define RULE_BUILDER_MULTI_HPP
#include "rule_builder_core.hpp"
#include "../rule_multi.hpp"


class RuleBuilderMulti : public RuleBuilderCore {
public:
    RuleMulti* create(Antecedent& antecedent) const override;
    RuleBuilderMulti* clone() const override;
};



#endif //RULE_BUILDER_MULTI_HPP
