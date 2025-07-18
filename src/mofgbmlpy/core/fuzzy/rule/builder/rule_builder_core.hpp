//
// Created by Robin on 16/07/2025.
//

#ifndef RULE_BUILDER_CORE_HPP
#define RULE_BUILDER_CORE_HPP
#include <memory>

#include "../../../data/dataset.hpp"
#include "../../../data/pattern.hpp"
#include "../abstract_rule.hpp"
#include "../antecedent/factory/abstract_antecedent_factory.hpp"
#include "../consequent/abstract_consequent.hpp"
#include "../consequent/learning/abstract_learning.hpp"


class RuleBuilderCore {
protected:
    std::shared_ptr<AbstractAntecedentFactory> antecedent_factory;
    std::shared_ptr<AbstractLearning> consequent_factory;
    std::shared_ptr<Knowledge> knowledge;

public:
    RuleBuilderCore(std::shared_ptr<AbstractAntecedentFactory> antecedent_factory,
                     std::shared_ptr<AbstractAntecedentFactory> consequent_factory,
                     std::shared_ptr<Knowledge> knowledge);
    RuleBuilderCore(const RuleBuilderCore& other);
    std::vector<std::vector<int>> create_antecedent_indices(const Pattern& pattern);
    std::vector<std::vector<int>> create_antecedent_indices(int num_rules=1);
    Antecedent* create_antecedent_from_indices(const std::vector<int>& antecedent_indices);
    std::shared_ptr<AbstractConsequent> create_consequent(Antecedent& antecedent, const Dataset& dataset);
    std::shared_ptr<AbstractConsequent> create_consequent(Antecedent& antecedent);
    std::shared_ptr<Knowledge> get_knowledge() const;
    std::shared_ptr<Dataset> get_training_dataset() const;
    std::shared_ptr<AbstractLearning> get_consequent_factory() const;

    virtual AbstractRule* create(Antecedent& antecedent) const = 0;

    virtual RuleBuilderCore* clone() const = 0;
};



#endif //RULE_BUILDER_CORE_HPP
