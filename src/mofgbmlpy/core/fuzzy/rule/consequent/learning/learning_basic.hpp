#ifndef LEARNING_BASIC_HPP
#define LEARNING_BASIC_HPP

#include <memory>
#include <vector>

#include "../../../../data/dataset.hpp"
#include "../../antecedent/antecedent.hpp"
#include "../abstract_consequent.hpp"
#include "abstract_learning.hpp"
#include "../../../../data/class_label/class_label_basic.hpp"
#include "../ruleWeight/rule_weight_basic.hpp"


class LearningBasic : public AbstractLearning
{
public:
    LearningBasic(const std::shared_ptr<Dataset>& training_dataset);
    LearningBasic(const LearningBasic& other);

    std::shared_ptr<AbstractConsequent> learning(
        Antecedent& antecedent,
        const Dataset& dataset,
        double reject_threshold = 0
    ) override;

    std::shared_ptr<AbstractConsequent> learning(
        Antecedent& antecedent,
        double reject_threshold = 0
    ) override;

    std::vector<double> calc_confidence(
        Antecedent& antecedent,
        const Dataset& dataset);

    std::vector<double> calc_confidence(
        Antecedent& antecedent);

    static std::shared_ptr<ClassLabelBasic> calc_class_label(const std::vector<double>& confidence);
    static std::shared_ptr<RuleWeightBasic> calc_rule_weight(ClassLabelBasic& class_label,
                                                             const std::vector<double>& confidence,
                                                             double reject_threshold);

    LearningBasic* clone() const override;
    operator std::string() const override;
    bool operator==(const AbstractLearning& other) const override;
};
#endif // LEARNING_BASIC_HPP
