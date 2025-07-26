#ifndef ABSTRACT_CONSEQUENT_HPP
#define ABSTRACT_CONSEQUENT_HPP

#include <memory>
#include <string>

#include "../../../data/class_label/abstract_class_label.hpp"
#include "ruleWeight/abstract_rule_weight.hpp"

class AbstractClassLabel;
class AbstractRuleWeight;

class AbstractConsequent {
public:
    virtual ~AbstractConsequent() = default;
    virtual AbstractClassLabel* get_class_label() const = 0;

    virtual bool is_rejected() const
    {
        return get_class_label()->is_rejected();
    }
    virtual void set_rejected()
    {
        get_class_label()->set_rejected();
    }

    virtual AbstractRuleWeight* get_rule_weight() const = 0;
    virtual void set_rule_weight(AbstractRuleWeight* rule_weight) = 0;

    virtual std::string get_linguistic_representation()
    {
        return "Class is " + std::string(*get_class_label()) + " with RW: " + std::string(*get_rule_weight());
    }

    virtual AbstractConsequent* clone() const = 0;
    virtual bool operator==(const AbstractConsequent& other) const = 0;
    virtual operator std::string() const {
        return "class: [" + std::string(*get_class_label()) + "]: weight: [" + std::string(*get_rule_weight()) + "]";
    }
    std::string to_string() const {
        return static_cast<std::string>(*this);
    }
};

#endif // ABSTRACT_CONSEQUENT_HPP
