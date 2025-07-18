//
// Created by Robin on 12/07/2025.
//

#ifndef MICHIGAN_SOLUTION_BUILDER_HPP
#define MICHIGAN_SOLUTION_BUILDER_HPP
#include "solution_builder_core.hpp"
#include "../../../fuzzy/rule/builder/rule_builder_core.hpp"
#include "../../../gbml/solution/michigan_solution.hpp"


class MichiganSolutionBuilder : public SolutionBuilderCore {
protected:
    std::shared_ptr<RandomGenerator> random_gen;
public:
    MichiganSolutionBuilder(std::shared_ptr<RandomGenerator> random_gen, int num_objectives, int num_constraints, std::shared_ptr<RuleBuilderCore> rule_builder);
    MichiganSolutionBuilder(const MichiganSolutionBuilder& other);
    std::vector<std::shared_ptr<MichiganSolution>> create(Pattern& pattern, int num_solutions=1);
    std::vector<std::shared_ptr<MichiganSolution>> create(int num_solutions=1);
    MichiganSolutionBuilder* clone() const override;
};


#endif //MICHIGAN_SOLUTION_BUILDER_HPP
