//
// Created by Robin on 16/07/2025.
//

#include "abstract_rule.hpp"

AbstractRule::AbstractRule(std::shared_ptr<Antecedent> antecedent, std::shared_ptr<AbstractConsequent> consequent)
    : antecedent(std::move(antecedent)), consequent(std::move(consequent))
{
    if (!this->antecedent || !this->consequent) {
        throw std::invalid_argument("Antecedent and consequent must not be null.");
    }
}

AbstractRule::AbstractRule(const AbstractRule& other)
{
    if (!this->antecedent || !this->consequent) {
        throw std::invalid_argument("Antecedent and consequent must not be null.");
    }

    Antecedent* antecedent_copy = other.antecedent->clone();
    this->antecedent = std::shared_ptr<Antecedent>(antecedent_copy);

    AbstractConsequent* consequent_copy = other.consequent->clone();
    this->consequent = std::shared_ptr<AbstractConsequent>(consequent_copy);
}

std::shared_ptr<Antecedent> AbstractRule::get_antecedent() const
{
    return antecedent;
}

std::shared_ptr<AbstractConsequent> AbstractRule::get_consequent() const
{
    return consequent;
}

void AbstractRule::set_consequent(std::shared_ptr<AbstractConsequent> consequent)
{
    this->consequent = std::move(consequent);
}

std::vector<double> AbstractRule::get_membership_values(const std::vector<double>& attribute_vector) const
{
    return antecedent->get_membership_values(attribute_vector);
}

double AbstractRule::get_compatible_grade_value(const std::vector<double>& attribute_vector) const
{
    return antecedent->get_compatible_grade_value(attribute_vector);
}

std::shared_ptr<AbstractClassLabel> AbstractRule::get_class_label() const
{
    return consequent->get_class_label();
}

bool AbstractRule::is_rejected_class_label() const
{
    return consequent->get_class_label()->is_rejected();
}

std::shared_ptr<AbstractRuleWeight> AbstractRule::get_rule_weight() const
{
    return consequent->get_rule_weight();
}

int AbstractRule::get_length() const
{
    return antecedent->get_length();
}

std::shared_ptr<Knowledge> AbstractRule::get_knowledge() const
{
    return antecedent->get_knowledge();
}

std::shared_ptr<FuzzySet> AbstractRule::get_fuzzy_set_object(int dim_index) const
{
    const int fuzzy_set_index = antecedent->get_antecedent_indices()[dim_index];
    return get_knowledge()->get_fuzzy_set(dim_index, fuzzy_set_index);
}

int AbstractRule::get_antecedent_array_size() const
{
    return antecedent->get_array_size();
}

std::string AbstractRule::get_var_name(int dim_index) const
{
    return get_knowledge()->get_fuzzy_variable(dim_index)->get_name();
}

std::string AbstractRule::get_linguistic_representation() const
{
    return "IF\t" + antecedent->get_linguistic_representation() +
           " THEN " + consequent->get_linguistic_representation() +
           " RW: " + std::string(*consequent->get_rule_weight());
}

AbstractRule::operator std::string() const
{
    return "Antecedent: " + std::string(*antecedent) + " => Consequent: " + std::string(*consequent);
}

bool AbstractRule::operator==(const AbstractRule& other) const
{
    return *antecedent == *other.antecedent && *consequent == *other.consequent;
}

std::string AbstractRule::to_string() const
{
    return static_cast<std::string>(*this);
}

