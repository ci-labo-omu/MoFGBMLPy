//
// Created by Robin on 16/07/2025.
//

#include "rule_builder_basic.hpp"

#include <memory>

#include "../consequent/consequent_basic.hpp"
#include "../consequent/learning/learning_basic.hpp"
#include "../rule_basic.hpp"

RuleBasic* RuleBuilderBasic::create(Antecedent& antecedent) const
{
    LearningBasic* learning_basic = dynamic_cast<LearningBasic*>(consequent_factory.get());
    ConsequentBasic* consequent = dynamic_cast<ConsequentBasic*>(learning_basic->learning(antecedent).get());
    return new RuleBasic(
        std::make_shared<Antecedent>(antecedent),
        std::make_shared<ConsequentBasic>(*consequent)
    );
}

RuleBuilderBasic* RuleBuilderBasic::clone() const
{
    return new RuleBuilderBasic(*this);
}
