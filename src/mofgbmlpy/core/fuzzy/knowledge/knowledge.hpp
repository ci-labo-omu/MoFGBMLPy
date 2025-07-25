#pragma once
#include <vector>
#include <memory>
#include "../fuzzy_term/fuzzy_variable.hpp"
#include "../fuzzy_term/fuzzy_set/fuzzy_set.hpp"

class Knowledge {
private:
    std::vector<FuzzyVariable*> fuzzy_vars;

public:
    Knowledge() = default;
    Knowledge(const std::vector<FuzzyVariable*>& fuzzy_vars);
    Knowledge(const Knowledge& other);

    FuzzyVariable* get_fuzzy_variable(int dim) const;
    FuzzySet* get_fuzzy_set(int dim, int fuzzy_set_index) const;
    int get_num_fuzzy_sets(int dim) const;
    void set_fuzzy_vars(const std::vector<FuzzyVariable*>& fuzzy_vars);
    const std::vector<FuzzyVariable*>& get_fuzzy_vars() const;
    float get_membership_value(double attribute_value, int dim, int fuzzy_set_index) const;
    int get_num_dim() const;
    float get_support(int dim, int fuzzy_set_index) const;
    Knowledge* clone() const;
    operator std::string() const;
    std::string to_string() const;
    bool operator==(const Knowledge& other) const;
};
