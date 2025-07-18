#ifndef LEARNING_MULTI_HPP
#define LEARNING_MULTI_HPP

#include <memory>
#include <vector>

#include "../../../../data/dataset.hpp"
#include "../../antecedent/antecedent.hpp"
#include "../abstract_consequent.hpp"
#include "abstract_learning.hpp"
#include "../../../../data/class_label/class_label_multi.hpp"
#include "../ruleWeight/rule_weight_multi.hpp"



class LearningMulti : public AbstractLearning
{
public:
    LearningMulti(const std::shared_ptr<Dataset>& training_dataset);
    LearningMulti(const LearningMulti& other);

    std::shared_ptr<AbstractConsequent> learning(
        Antecedent& antecedent,
        const Dataset& dataset,
        double reject_threshold = 0
    ) override;

    std::shared_ptr<AbstractConsequent> learning(
        Antecedent& antecedent,
        double reject_threshold = 0
    ) override;

    std::vector<std::vector<double>> calc_confidence(
        Antecedent& antecedent,
        const Dataset& dataset);

    std::vector<std::vector<double>> calc_confidence(
        Antecedent& antecedent);

    std::shared_ptr<ClassLabelMulti> calc_class_label(const std::vector<std::vector<double>>& confidence);
    std::shared_ptr<RuleWeightMulti> calc_rule_weight(ClassLabelMulti& class_label,
                                     const std::vector<std::vector<double>>& confidence,
                                     double reject_threshold);

    LearningMulti* clone() const override;
    operator std::string() const override;
    bool operator==(const AbstractLearning& other) const override;
};


#endif // LEARNING_MULTI_HPP
