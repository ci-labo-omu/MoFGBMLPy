//
// Created by Robin on 16/07/2025.
//

#include "rule_builder_core.hpp"

#include "../antecedent/factory/heuristic_antecedent_factory.hpp"

RuleBuilderCore::RuleBuilderCore(std::shared_ptr<AbstractAntecedentFactory> antecedent_factory,
                                 std::shared_ptr<AbstractAntecedentFactory> consequent_factory, std::shared_ptr<Knowledge> knowledge) {}

RuleBuilderCore::RuleBuilderCore(const RuleBuilderCore& other)
    : antecedent_factory(other.antecedent_factory),
      consequent_factory(other.consequent_factory),
      knowledge(other.knowledge) {}

std::vector<std::vector<int>> RuleBuilderCore::create_antecedent_indices(const Pattern& pattern)
{
    auto* heuristic_antecedent_factory = dynamic_cast<HeuristicAntecedentFactory*>(this->antecedent_factory.get());
    if (heuristic_antecedent_factory == nullptr) {
        throw std::runtime_error("Antecedent factory is not of type HeuristicAntecedentFactory");
    }
    return heuristic_antecedent_factory->create_antecedent_indices_from_pattern(pattern);
}

std::vector<std::vector<int>> RuleBuilderCore::create_antecedent_indices(const int num_rules)
{
    return antecedent_factory->create_antecedent_indices(num_rules);
}

Antecedent* RuleBuilderCore::create_antecedent_from_indices(const std::vector<int>& antecedent_indices)
{
    return new Antecedent(antecedent_indices, antecedent_factory->get_knowledge());
}

std::shared_ptr<AbstractConsequent> RuleBuilderCore::create_consequent(Antecedent& antecedent, const Dataset& dataset)
{
    return consequent_factory->learning(antecedent, dataset);
}

std::shared_ptr<AbstractConsequent> RuleBuilderCore::create_consequent(Antecedent& antecedent)
{
    return consequent_factory->learning(antecedent);
}

std::shared_ptr<Knowledge> RuleBuilderCore::get_knowledge() const
{
    return knowledge;
}

std::shared_ptr<Dataset> RuleBuilderCore::get_training_dataset() const
{
    return consequent_factory->get_training_set();
}

std::shared_ptr<AbstractLearning> RuleBuilderCore::get_consequent_factory() const
{
    return consequent_factory;
}
