#ifndef SINGLE_WINNER_RULE_SELECTION_HPP
#define SINGLE_WINNER_RULE_SELECTION_HPP

#include "abstract_classification.hpp"
#include <vector>
#include <memory>

// Forward declarations
class Pattern;
class MichiganSolution;

class SingleWinnerRuleSelection : public AbstractClassification
{
public:
    SingleWinnerRuleSelection() = default;
    ~SingleWinnerRuleSelection() = default;

    std::shared_ptr<MichiganSolution> classify(
        const std::vector<std::shared_ptr<MichiganSolution>>& michigan_solution_list,
        const Pattern& pattern) override;

    AbstractClassification* clone() const override;
    bool operator==(const AbstractClassification& other) const override;
    operator std::string() const override;
};

#endif // SINGLE_WINNER_RULE_SELECTION_HPP