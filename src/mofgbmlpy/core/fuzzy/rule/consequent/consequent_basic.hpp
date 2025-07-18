#ifndef CONSEQUENT_BASIC_HPP
#define CONSEQUENT_BASIC_HPP

#include "abstract_consequent.hpp"
#include "../../../data/class_label/class_label_basic.hpp"
#include "ruleWeight/rule_weight_basic.hpp"


class ConsequentBasic : public AbstractConsequent {
private:
    std::shared_ptr<ClassLabelBasic> class_label;
    std::shared_ptr<RuleWeightBasic> rule_weight;

public:
    ConsequentBasic(std::shared_ptr<ClassLabelBasic> class_label,
                    std::shared_ptr<RuleWeightBasic> rule_weight);
    ConsequentBasic(const ConsequentBasic& other);

    std::shared_ptr<AbstractClassLabel> get_class_label() const override;
    void set_class_label_value(int value);
    int get_class_label_value() const;

    std::shared_ptr<AbstractRuleWeight> get_rule_weight() const override;
    void set_rule_weight(std::shared_ptr<AbstractRuleWeight> rule_weight) override;

    ConsequentBasic* clone() const override;
    bool operator==(const AbstractConsequent& other) const override;
};


#endif  // CONSEQUENT_BASIC_HPP
