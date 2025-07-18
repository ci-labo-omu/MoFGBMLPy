#include "learning_basic.hpp"
#include <cmath>

#include "../consequent_basic.hpp"


LearningBasic::LearningBasic(const std::shared_ptr<Dataset>& training_dataset)
    : AbstractLearning(training_dataset)
{
}

LearningBasic::LearningBasic(const LearningBasic& other)
    : AbstractLearning(other.get_training_set())
{
}

std::shared_ptr<AbstractConsequent> LearningBasic::learning(Antecedent& antecedent, const Dataset& dataset,
                                                            double reject_threshold)
{
    auto confidence = calc_confidence(antecedent, dataset);
    auto class_label = calc_class_label(confidence);
    auto rule_weight = calc_rule_weight(*class_label, confidence, reject_threshold);
    return std::make_shared<ConsequentBasic>(class_label, rule_weight);
}

std::shared_ptr<AbstractConsequent> LearningBasic::learning(Antecedent& antecedent, double reject_threshold)
{
    auto confidence = calc_confidence(antecedent);
    auto class_label = calc_class_label(confidence);
    auto rule_weight = calc_rule_weight(*class_label, confidence, reject_threshold);
    return std::make_shared<ConsequentBasic>(class_label, rule_weight);
}

std::vector<double> LearningBasic::calc_confidence(
    Antecedent& antecedent,
    const Dataset& dataset)
{
    int num_classes = dataset.get_num_classes();
    int dataset_size = dataset.get_size();

    std::vector<double> confidence(num_classes, 0.0);
    std::vector<double> sum_compatible_grade_for_each_class(num_classes, 0.0);

    double all_sum = 0.0;
    for (int i = 0; i < dataset_size; i++) {
        auto pattern = dataset.get_pattern(i);
        double compatible_grade = antecedent.get_compatible_grade_value(pattern->get_attributes_vector());
        ClassLabelBasic* class_label_ptr = dynamic_cast<ClassLabelBasic*>(pattern->get_target_class().get());
        int class_label = class_label_ptr->get_class_label_value();

        sum_compatible_grade_for_each_class[class_label] += compatible_grade;
        all_sum += compatible_grade;
    }

    if (all_sum != 0.0) {
        for (int i = 0; i < num_classes; i++) {
            confidence[i] = sum_compatible_grade_for_each_class[i] / all_sum;
        }
    }

    return confidence;
}

std::vector<double> LearningBasic::calc_confidence(Antecedent& antecedent)
{
    return calc_confidence(antecedent, *train_ds);
}

std::shared_ptr<ClassLabelBasic> LearningBasic::calc_class_label(const std::vector<double>& confidence)
{
    double max_val = -INFINITY;
    int consequent_class = -1;

    for (int i = 0; i < confidence.size(); i++) {
        if (confidence[i] > max_val) {
            max_val = confidence[i];
            consequent_class = i;
        }
        else if (confidence[i] == max_val) {
            consequent_class = -1;
        }
    }

    if (consequent_class < 0) {
        std::shared_ptr<ClassLabelBasic> class_label = std::make_shared<ClassLabelBasic>(-1);
        class_label->set_rejected();
        return class_label;
    }
    return std::make_shared<ClassLabelBasic>(consequent_class);
}

std::shared_ptr<RuleWeightBasic> LearningBasic::calc_rule_weight(
    ClassLabelBasic& class_label,
    const std::vector<double>& confidence,
    double reject_threshold)
{
    if (class_label.is_rejected()) {
        return std::make_shared<RuleWeightBasic>(0.0);
    }

    int label_value = class_label.get_class_label_value();
    if (label_value < 0 || label_value >= confidence.size()) {
        throw std::out_of_range("Label value out of confidence bounds");
    }

    // # cdef double sum_confidence = np.sum(confidence, dtype=np.float64)
    // # cdef double rule_weight_val = confidence[label_value] - (sum_confidence - confidence[label_value])
    // # TODO Re-check the effect of this modification on the results and recheck it's validity
    // # It seems in the java version that the sum is 1, but here (due to imprecision probably) it can be slightly different
    double rule_weight_val = (confidence[label_value] * 2.0) - 1.0;

    if (rule_weight_val <= reject_threshold) {
        class_label.set_rejected();
        return std::make_shared<RuleWeightBasic>(0.0);
    }

    return std::make_shared<RuleWeightBasic>(rule_weight_val);
}

LearningBasic* LearningBasic::clone() const
{
    return new LearningBasic(*this);
}

LearningBasic::operator std::string() const
{
    return "MoFGBML_Learning";
}

bool LearningBasic::operator==(const AbstractLearning& other) const
{
    const LearningBasic* other_learning = dynamic_cast<const LearningBasic*>(&other);
    if (!other_learning) {
        return false;
    }
    return *train_ds == *(other_learning->get_training_set());
}


