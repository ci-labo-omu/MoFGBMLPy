#ifndef MOFGBMLPY_CORE_FUZZY_TERM_FUZZY_VARIABLE_HPP
#define MOFGBMLPY_CORE_FUZZY_TERM_FUZZY_VARIABLE_HPP

#include <string>
#include <vector>
#include <memory>
#include "fuzzy_set/fuzzy_set.hpp"

class FuzzyVariable {
private:
    std::vector<std::shared_ptr<FuzzySet>> fuzzy_sets;
    std::string name;
    std::vector<float> domain;
public:
    FuzzyVariable(const std::vector<std::shared_ptr<FuzzySet>>& fuzzy_sets, const std::string& name = "unamed_var", const std::vector<float>& domain = {0.0f, 1.0f});
    FuzzyVariable(const FuzzyVariable& other);
    std::string get_name() const;
    float get_membership_value(int fuzzy_set_index, float x) const;
    int get_length() const;
    std::shared_ptr<FuzzySet> get_fuzzy_set(int fuzzy_set_index) const;
    float get_support(int fuzzy_set_index) const;
    std::vector<std::shared_ptr<FuzzySet>> get_fuzzy_sets() const;
    std::vector<float> get_support_values() const;
    std::vector<float> get_domain() const;
    FuzzyVariable* clone() const;
    operator std::string() const;
    std::string to_string() const;
    bool operator==(const FuzzyVariable& other) const;
};

#endif // MOFGBMLPY_CORE_FUZZY_TERM_FUZZY_VARIABLE_HPP

