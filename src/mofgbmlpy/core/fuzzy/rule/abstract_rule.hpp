//
// Created by Robin on 16/07/2025.
//

#ifndef ABSTRACT_RULE_HPP
#define ABSTRACT_RULE_HPP

#include "antecedent/antecedent.hpp"
#include "consequent/abstract_consequent.hpp"

class AbstractRule {
protected:
    std::shared_ptr<Antecedent> antecedent;
    std::shared_ptr<AbstractConsequent> consequent;

public:
    AbstractRule(std::shared_ptr<Antecedent> antecedent, std::shared_ptr<AbstractConsequent> consequent);
    AbstractRule(const AbstractRule& other);
    virtual ~AbstractRule() = default;

    virtual std::shared_ptr<Antecedent> get_antecedent() const;
    virtual std::shared_ptr<AbstractConsequent> get_consequent() const;
    virtual void set_consequent(std::shared_ptr<AbstractConsequent> consequent);
    virtual std::vector<double> get_membership_values(const std::vector<double>& attribute_vector) const;
    virtual double get_compatible_grade_value(const std::vector<double>& attribute_vector) const;
    virtual std::shared_ptr<AbstractClassLabel> get_class_label() const;
    virtual bool is_rejected_class_label() const ;
    virtual std::shared_ptr<AbstractRuleWeight> get_rule_weight() const;
    virtual int get_length() const;
    virtual double get_fitness_value(const std::vector<double>& attribute_vector) const = 0;
    virtual std::shared_ptr<Knowledge> get_knowledge() const;
    virtual std::shared_ptr<FuzzySet> get_fuzzy_set_object(int dim_index) const;
    virtual int get_antecedent_array_size() const;
    virtual std::string get_var_name(int dim_index) const;
    virtual std::string get_linguistic_representation() const;

    virtual operator std::string() const;
    virtual bool operator==(const AbstractRule& other) const;
    virtual AbstractRule* clone() const = 0;
    std::string to_string() const;
};



#endif //ABSTRACT_RULE_HPP
