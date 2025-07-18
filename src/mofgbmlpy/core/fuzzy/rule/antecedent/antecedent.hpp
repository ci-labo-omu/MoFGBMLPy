
#ifndef MOFGBMLPY_FUZZY_RULE_ANTECEDENT_ANTECEDENT_HPP
#define MOFGBMLPY_FUZZY_RULE_ANTECEDENT_ANTECEDENT_HPP

#include <vector>
#include <string>
#include "../../knowledge/knowledge.hpp"

class Antecedent {
private:
    std::vector<int> antecedent_indices;
    std::shared_ptr<Knowledge> knowledge;

public:
    Antecedent(std::vector<int> antecedent_indices, const std::shared_ptr<Knowledge>& knowledge);
    Antecedent(const Antecedent& other);

    int get_array_size() const;
    std::vector<int> get_antecedent_indices() const;
    void set_antecedent_indices(const std::vector<int>& new_indices);
    std::vector<double> get_membership_values(const std::vector<double>& attribute_vector) const;
    double get_compatible_grade_value(const std::vector<double>& attribute_vector) const;
    int get_length() const;
    std::string get_linguistic_representation() const;
    std::shared_ptr<Knowledge> get_knowledge() const;
    void set_knowledge(Knowledge* new_knowledge);

    operator std::string() const;
    std::string to_string() const;
    bool operator==(const Antecedent& other) const;
    Antecedent* clone() const;
};

#endif
