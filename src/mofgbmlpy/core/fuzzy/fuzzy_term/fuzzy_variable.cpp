#include "fuzzy_variable.hpp"
#include <stdexcept>

FuzzyVariable::FuzzyVariable(const std::vector<FuzzySet*>& fuzzy_sets, const std::string& name, const std::vector<float>& domain)
    : fuzzy_sets(fuzzy_sets), name(name), domain(domain) {
    if (name.empty()) {
        throw std::invalid_argument("name can't be empty");
    }
    if (fuzzy_sets.empty()) {
        throw std::invalid_argument("fuzzy_sets must have at least one element");
    }
    if (domain.size() != 2) {
        throw std::invalid_argument("domain must have exactly two elements");
    }

    if (domain[0] >= domain[1]) {
        throw std::invalid_argument("domain[0] must be less than domain[1]");
    }
}

FuzzyVariable::FuzzyVariable(const FuzzyVariable& other) {
    name = other.name;
    domain = std::vector<float>(other.domain);

    fuzzy_sets = std::vector<FuzzySet*>(other.fuzzy_sets.size());
    for (size_t i = 0; i < other.fuzzy_sets.size(); ++i) {
        fuzzy_sets[i] = other.fuzzy_sets[i]->clone();
    }
}

std::string FuzzyVariable::get_name() const {
    return name;
}

float FuzzyVariable::get_membership_value(int fuzzy_set_index, float x) const {
    return get_fuzzy_set(fuzzy_set_index)->get_membership_value(x);
}

int FuzzyVariable::get_length() const {
    return static_cast<int>(fuzzy_sets.size());
}

FuzzySet* FuzzyVariable::get_fuzzy_set(int fuzzy_set_index) const {
    return fuzzy_sets.at(fuzzy_set_index);
}

float FuzzyVariable::get_support(int fuzzy_set_index) const {
    return get_fuzzy_set(fuzzy_set_index)->get_support();
}

std::vector<FuzzySet*> FuzzyVariable::get_fuzzy_sets() const {
    return fuzzy_sets;
}

std::vector<float> FuzzyVariable::get_support_values() const {
    std::vector<float> supports = std::vector<float>(fuzzy_sets.size());
    for (int i = 0; i < static_cast<int>(fuzzy_sets.size()); ++i) {
        supports[i] = fuzzy_sets[i]->get_support();
    }
    return supports;
}

std::vector<float> FuzzyVariable::get_domain() const {
    return domain;
}

FuzzyVariable* FuzzyVariable::clone() const {
    return new FuzzyVariable(*this);
}

FuzzyVariable::operator std::string() const {
    std::string result = "FuzzyVariable: " + name + "\n";
    for (FuzzySet* fs : fuzzy_sets) {
        result += "\t" + std::string(*fs) + "\n";
    }
    return result;
}

bool FuzzyVariable::operator==(const FuzzyVariable& other) const {
    if (!(name == other.name && domain == other.domain)) {
        return false;
    }

    if (get_length() != other.get_length()) {
        return false;
    }

    for (int i = 0; i < get_length(); ++i) {
        if (!(*(fuzzy_sets[i]) == *(other.fuzzy_sets[i]))) {
            return false;
        }
    }

    return true;
}

std::string FuzzyVariable::to_string() const {
    return static_cast<std::string>(*this);
}
