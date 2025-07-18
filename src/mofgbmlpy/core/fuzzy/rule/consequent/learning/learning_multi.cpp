#include "learning_multi.hpp"
#include <cmath> // INFINITY
#include <stdexcept>

#include "learning_basic.hpp"
#include "../consequent_multi.hpp"


LearningMulti::LearningMulti(const std::shared_ptr<Dataset>& training_dataset)
    : AbstractLearning(training_dataset) {}

LearningMulti::LearningMulti(const LearningMulti& other)
    : AbstractLearning(other.get_training_set()) {}

std::shared_ptr<AbstractConsequent> LearningMulti::learning(Antecedent& antecedent, double reject_threshold)
{
    return AbstractLearning::learning(antecedent, reject_threshold);
}

std::shared_ptr<AbstractConsequent> LearningMulti::learning(Antecedent& antecedent, const Dataset& dataset, double reject_threshold) {
    auto confidence = calc_confidence(antecedent, dataset);
    auto class_label = calc_class_label(confidence);
    auto rule_weight = calc_rule_weight(*class_label, confidence, reject_threshold);
    return std::make_shared<ConsequentMulti>(class_label, rule_weight);
}

std::vector<std::vector<double>> LearningMulti::calc_confidence(Antecedent& antecedent, const Dataset& dataset) {
    int num_classes = dataset.get_num_classes();
    int dataset_size = dataset.get_size();

    // Initialize confidence matrix: num_classes x 2 (OFF, ON)
    std::vector<std::vector<double>> confidence(num_classes, std::vector<double>(2, 0.0));
    std::vector<double> compatible_grades(dataset_size, 0.0);

    // Cache compatible grades for all patterns
    for (int i = 0; i < dataset_size; ++i) {
        auto pattern = dataset.get_pattern(i);
        compatible_grades[i] = antecedent.get_compatible_grade_value(pattern->get_attributes_vector());
    }

    for (int c = 0; c < num_classes; ++c) {
        for (int i = 0; i < dataset_size; ++i) {
            auto pattern = dataset.get_pattern(i);
            ClassLabelMulti* class_label_ptr = dynamic_cast<ClassLabelMulti*>(pattern->get_target_class().get());
            int class_label_val = class_label_ptr->get_class_label_value_at(c);
            confidence[c][class_label_val] += compatible_grades[i];
        }
        double all_sum = confidence[c][0] + confidence[c][1];
        if (all_sum != 0) {
            confidence[c][0] /= all_sum;
            confidence[c][1] /= all_sum;
        } else {
            confidence[c][0] = 0.0;
            confidence[c][1] = 0.0;
        }
    }

    return confidence;
}

std::vector<std::vector<double>> LearningMulti::calc_confidence(Antecedent& antecedent)
{
    return calc_confidence(antecedent, *train_ds);
}


std::shared_ptr<ClassLabelMulti> LearningMulti::calc_class_label(const std::vector<std::vector<double>>& confidence)
{
    int n = static_cast<int>(confidence.size());
    std::vector<int> consequent_classes(n, -1);

    for (int c = 0; c < n; ++c) {
        if (confidence[c][0] > confidence[c][1]) {
            consequent_classes[c] = 0;
        } else if (confidence[c][0] < confidence[c][1]) {
            consequent_classes[c] = 1;
        } else {
            std::shared_ptr<ClassLabelMulti> class_label = std::make_shared<ClassLabelMulti>(consequent_classes);
            class_label->set_rejected();
            return class_label;
        }
    }
    return std::make_shared<ClassLabelMulti>(consequent_classes);
}

std::shared_ptr<RuleWeightMulti> LearningMulti::calc_rule_weight(ClassLabelMulti& class_label,
                                                const std::vector<std::vector<double>>& confidence,
                                                double reject_threshold)
{
    if (class_label.is_rejected()) {
        return std::make_shared<RuleWeightMulti>(std::vector<double>(confidence.size(), 0.0));
    }

    int n = static_cast<int>(confidence.size());
    std::vector<double> rule_weight_values(n, -1.0);

    for (int c = 0; c < n; ++c) {
        rule_weight_values[c] = std::abs(confidence[c][0] - confidence[c][1]);
    }

    return std::make_shared<RuleWeightMulti>(rule_weight_values);
}

LearningMulti* LearningMulti::clone() const
{
    return new LearningMulti(*this);
}

LearningMulti::operator std::string() const
{
    return "MoFGBML_Learning";
}

bool LearningMulti::operator==(const AbstractLearning& other) const
{
    const LearningMulti* other_multi = dynamic_cast<const LearningMulti*>(&other);
    if (!other_multi) {
        return false;
    }
    return *train_ds == *(other_multi->get_training_set());
}

