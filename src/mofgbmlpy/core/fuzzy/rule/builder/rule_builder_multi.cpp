//
// Created by Robin on 16/07/2025.
//

#include "rule_builder_multi.hpp"

#include "../rule_multi.hpp"
#include "../consequent/consequent_multi.hpp"
#include "../consequent/learning/learning_multi.hpp"

RuleMulti* RuleBuilderMulti::create(Antecedent& antecedent) const
{
    LearningMulti* learning_basic = dynamic_cast<LearningMulti*>(consequent_factory.get());
    ConsequentMulti* consequent = dynamic_cast<ConsequentMulti*>(learning_basic->learning(antecedent).get());

    return new RuleMulti(
        std::make_shared<Antecedent>(antecedent),
        std::make_shared<ConsequentMulti>(*consequent)
    );
}

RuleBuilderMulti* RuleBuilderMulti::clone() const
{
    return new RuleBuilderMulti(*this);
}
