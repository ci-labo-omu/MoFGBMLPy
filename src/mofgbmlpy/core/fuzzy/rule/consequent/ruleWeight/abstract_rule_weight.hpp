#ifndef ABSTRACT_RULE_WEIGHT_HPP
#define ABSTRACT_RULE_WEIGHT_HPP

#include <any>
#include <memory>
#include <variant>
#include <vector>
#include <string>


class AbstractRuleWeight {
public:
    virtual ~AbstractRuleWeight() = default;

    virtual std::variant<double, std::vector<double>> get_value() const = 0;
    virtual void set_value(std::variant<double, std::vector<double>> rule_weight) = 0;

    virtual operator std::string() const = 0;
    virtual bool operator==(const AbstractRuleWeight& other) const = 0;
    virtual AbstractRuleWeight* clone() const = 0;
    virtual std::string to_string() const {
        return static_cast<std::string>(*this);
    }
};


#endif  // ABSTRACT_RULE_WEIGHT_HPP
