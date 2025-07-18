//
// Created by Robin on 12/07/2025.
//

#ifndef SOLUTION_BUILDER_CORE_HPP
#define SOLUTION_BUILDER_CORE_HPP
#include <memory>
#include <utility>

#include "../../../fuzzy/rule/builder/rule_builder_core.hpp"


class SolutionBuilderCore {
protected:
    int num_objectives;
    int num_constraints;
    std::shared_ptr<RuleBuilderCore> rule_builder;

public:
    SolutionBuilderCore(int num_objectives, int num_constraints, std::shared_ptr<RuleBuilderCore> rule_builder)
        : num_objectives(num_objectives), num_constraints(num_constraints), rule_builder(std::move(rule_builder)) {}

    std::shared_ptr<RuleBuilderCore> get_rule_builder() const {
        return rule_builder;
    }

    virtual SolutionBuilderCore* clone() const = 0;
};



#endif //SOLUTION_BUILDER_CORE_HPP
