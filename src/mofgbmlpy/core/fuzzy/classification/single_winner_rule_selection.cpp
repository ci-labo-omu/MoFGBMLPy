#include "single_winner_rule_selection.hpp"
#include <limits>
#include <stdexcept>

std::shared_ptr<MichiganSolution> SingleWinnerRuleSelection::classify(
    const std::vector<std::shared_ptr<MichiganSolution>>& michigan_solution_list,
    const Pattern& pattern)
{
    double max = std::numeric_limits<double>::lowest();
    bool can_classify = false;

    if (michigan_solution_list.empty()) {
        throw std::runtime_error("No solutions available for classification.");
    }

    std::shared_ptr<MichiganSolution> winner = michigan_solution_list[0];

    for (const auto& solution : michigan_solution_list) {
        if(solution->is_rejected())
            throw std::runtime_error("One michigan solution has a rejected class label (it can't be used for classification)");

        double value = solution->get_fitness_value(pattern.get_attributes_vector());

        if (value > max) {
            max = value;
            winner = solution;
            can_classify = true;
        } else if (value == max && solution->get_class_label() != winner->get_class_label()) {
            // There are 2 best solutions with the same fitness value
            can_classify = false;
        }
    }

    if (can_classify && max >= 0)
        return winner;
    return nullptr;
}

AbstractClassification* SingleWinnerRuleSelection::clone() const
{
    return new SingleWinnerRuleSelection();
}

SingleWinnerRuleSelection::operator std::string() const
{
    return "SingleWinnerRuleSelection";
}

bool SingleWinnerRuleSelection::operator==(const AbstractClassification& other) const
{
    return dynamic_cast<const SingleWinnerRuleSelection*>(&other) != nullptr;
}
