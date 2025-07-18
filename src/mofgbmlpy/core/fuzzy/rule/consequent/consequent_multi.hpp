#ifndef CONSEQUENT_MULTI_HPP
#define CONSEQUENT_MULTI_HPP

#include "abstract_consequent.hpp"
#include "../../../data/class_label/class_label_multi.hpp"
#include "ruleWeight/rule_weight_multi.hpp"


class ConsequentMulti : public AbstractConsequent {
private:
    std::shared_ptr<ClassLabelMulti> class_label;
    std::shared_ptr<RuleWeightMulti> rule_weight;

public:
    ConsequentMulti(std::shared_ptr<ClassLabelMulti> class_label,
                    std::shared_ptr<RuleWeightMulti> rule_weight);
    ConsequentMulti(const ConsequentMulti& other);

    std::shared_ptr<AbstractClassLabel> get_class_label() const override;
    void set_class_label_value(std::vector<int> class_label_value);
    std::vector<int> get_class_label_value() const;

    std::shared_ptr<AbstractRuleWeight> get_rule_weight() const override;
    void set_rule_weight(std::shared_ptr<AbstractRuleWeight> rule_weight) override;

    ConsequentMulti* clone() const override;
    bool operator==(const AbstractConsequent& other) const override;
};


#endif  // CONSEQUENT_MULTI_HPP
