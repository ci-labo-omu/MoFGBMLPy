#include "knowledge.hpp"
#include <stdexcept>
#include <sstream>

Knowledge::Knowledge(const std::vector<FuzzyVariable*>& fuzzy_vars)
    : fuzzy_vars(fuzzy_vars) {
}

Knowledge::Knowledge(const Knowledge& other) {
    fuzzy_vars = std::vector<FuzzyVariable*>(other.fuzzy_vars.size());
    for(int i = 0; i < static_cast<int>(other.fuzzy_vars.size()); i++){
        fuzzy_vars[i] = (other.fuzzy_vars[i])->clone();
    }
}

FuzzyVariable* Knowledge::get_fuzzy_variable(int dim) const {
    return fuzzy_vars.at(dim);
}

FuzzySet* Knowledge::get_fuzzy_set(int dim, int fuzzy_set_index) const {
    if (fuzzy_vars.empty()) {
        throw std::runtime_error("Fuzzy variables are not initialized");
    }
    
    return fuzzy_vars.at(dim)->get_fuzzy_set(fuzzy_set_index);
}

int Knowledge::get_num_fuzzy_sets(int dim) const {
    if (fuzzy_vars.empty()) {
        throw std::runtime_error("Fuzzy variables are not initialized");
    }

    return fuzzy_vars.at(dim)->get_length();
}

void Knowledge::set_fuzzy_vars(const std::vector<FuzzyVariable*>& new_fuzzy_vars) {
    this->fuzzy_vars = new_fuzzy_vars;
}

const std::vector<FuzzyVariable*>& Knowledge::get_fuzzy_vars() const {
    return fuzzy_vars;
}

float Knowledge::get_membership_value(double attribute_value, int dim, int fuzzy_set_index) const {
    if (fuzzy_vars.empty()) {
        throw std::runtime_error("Fuzzy variables are not initialized");
    }
    
    return fuzzy_vars.at(dim)->get_membership_value(fuzzy_set_index, static_cast<float>(attribute_value));
}

int Knowledge::get_num_dim() const {
    return static_cast<int>(fuzzy_vars.size());
}

float Knowledge::get_support(int dim, int fuzzy_set_index) const {
    return fuzzy_vars.at(dim)->get_support(fuzzy_set_index);
}

Knowledge::operator std::string() const {
    std::ostringstream oss;
    for (int i = 0; i < get_num_dim(); ++i) {
        oss << std::string(*fuzzy_vars[i]) << "\n";
    }
    return oss.str();
}

std::string Knowledge::to_string() const {
    return static_cast<std::string>(*this);
}

bool Knowledge::operator==(const Knowledge& other) const {
    if (fuzzy_vars.size() != other.fuzzy_vars.size()) {
        return false;
    }

    for (size_t i = 0; i < fuzzy_vars.size(); ++i) {
        if (!(*fuzzy_vars[i] == *other.fuzzy_vars[i])) {
            return false;
        }
    }

    return true;
}

Knowledge* Knowledge::clone() const {
    return new Knowledge(*this);
}