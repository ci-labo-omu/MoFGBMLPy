//
// Created by Robin on 12/07/2025.
//

#include "michigan_solution_builder.hpp"

MichiganSolutionBuilder::MichiganSolutionBuilder(std::shared_ptr<RandomGenerator> random_gen, int num_objectives,
    int num_constraints, std::shared_ptr<RuleBuilderCore> rule_builder)
    : SolutionBuilderCore(num_objectives, num_constraints, rule_builder), random_gen(random_gen) {}

MichiganSolutionBuilder::MichiganSolutionBuilder(const MichiganSolutionBuilder& other)
    : MichiganSolutionBuilder(other.random_gen, other.num_objectives, other.num_constraints, std::shared_ptr<RuleBuilderCore>(other.rule_builder->clone())) {}

std::vector<std::shared_ptr<MichiganSolution>> MichiganSolutionBuilder::create(Pattern& pattern, int num_solutions)
{
    std::vector<std::shared_ptr<MichiganSolution>> solutions;
    solutions.reserve(num_solutions);
    for (int i = 0; i < num_solutions; i++) {
        MichiganSolution* solution = new MichiganSolution(random_gen, num_objectives, num_constraints, rule_builder, pattern);
        solutions.push_back(std::shared_ptr<MichiganSolution>(solution));
    }
    return solutions;
}

std::vector<std::shared_ptr<MichiganSolution>> MichiganSolutionBuilder::create(int num_solutions)
{
    std::vector<std::shared_ptr<MichiganSolution>> solutions;
    solutions.reserve(num_solutions);
    for (int i = 0; i < num_solutions; i++) {
        MichiganSolution* solution = new MichiganSolution(random_gen, num_objectives, num_constraints, rule_builder);
        solutions.push_back(std::shared_ptr<MichiganSolution>(solution));
    }
    return solutions;
}

MichiganSolutionBuilder* MichiganSolutionBuilder::clone() const
{
    return new MichiganSolutionBuilder(*this);
}
