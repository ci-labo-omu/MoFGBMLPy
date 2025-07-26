#include "antecedent.hpp"
#include <sstream>
#include <stdexcept>
#include <cmath>
#include <iostream>
#include <string>

Antecedent::Antecedent(std::vector<int>& antecedent_indices, Knowledge* knowledge)
: antecedent_indices(antecedent_indices), knowledge(knowledge) {
    if (knowledge == nullptr)
        throw std::invalid_argument("Knowledge cannot be nullptr");

    if(antecedent_indices.empty())
        throw std::invalid_argument("Antecedent indices cannot be empty");

    if (antecedent_indices.size() != knowledge->get_num_dim())
        throw std::invalid_argument("The number of antecedent indices must match the number of dimensions in knowledge");
}

Antecedent::Antecedent(const Antecedent& other) : antecedent_indices(other.antecedent_indices), knowledge(other.knowledge) {}

int Antecedent::get_array_size() const {
    return static_cast<int>(antecedent_indices.size());
}

std::vector<int> Antecedent::get_antecedent_indices() const {
    return antecedent_indices;
}

void Antecedent::set_antecedent_indices(const std::vector<int>& new_indices) {
    if (new_indices.size() > knowledge->get_num_dim())
        throw std::invalid_argument("The given number of dimensions is out of bounds for the current knowledge");
    antecedent_indices = new_indices;
}

std::vector<double> Antecedent::get_membership_values(const std::vector<double>& attribute_vector) const {
    int size = this->get_array_size();
    if (attribute_vector.size() != size)
        throw std::invalid_argument("attribute_vector length mismatch");

    std::vector<double> grade(size, 0.0);

    for (int i = 0; i < size; i++) {
        int idx = antecedent_indices[i];
        double val = attribute_vector[i];

        if (idx < 0 && val < 0) {
            // categorical
            grade[i] = (idx == std::round(val)) ? 1.0 : 0.0;
        }
        else if (idx > 0 && val >= 0) {
            // numerical
            grade[i] = knowledge->get_membership_value(val, i, idx);
        }
        else if (idx == 0) {
            // don't care
            grade[i] = 1.0;
        }
        else {
            throw std::invalid_argument("Incompatible antecedent index with input: dimension=" + std::to_string(i) + ", value=" + std::to_string(val) + ", index=" + std::to_string(idx));
        }
    }
    return grade;
}

double Antecedent::get_compatible_grade_value(const std::vector<double>& attribute_vector) const {
    int size = this->get_array_size();
    if (attribute_vector.size() != size)
        throw std::invalid_argument("attribute_vector length mismatch");

    double grade_value = 1.0;

    for (int i = 0; i < size; i++) {
        int idx = antecedent_indices[i];
        if (idx == 0) continue; // Don't care skip

        double val = attribute_vector[i];

        if (idx < 0) {
            // Categorical
            if (val < 0) {
                if (idx != static_cast<int>(std::round(val)))
                    return 0.0;
            }
            else {
                throw std::invalid_argument("Incompatible antecedent index with input: dimension=" + std::to_string(static_cast<int>(i)) + ", value=" + std::to_string(val) + ", index=" + std::to_string(idx));
            }
        }
        else {
            // Numerical
            if (val >= 0) {
                double membership_val = knowledge->get_membership_value(val, static_cast<int>(i), idx);
                if (membership_val == 0.0)
                    return 0.0;
                grade_value *= membership_val;
            }
            else {
                throw std::invalid_argument("Incompatible antecedent index with input: dimension=" + std::to_string(static_cast<int>(i)) + ", value=" + std::to_string(val) + ", index=" + std::to_string(idx));
            }
        }
    }
    return grade_value;
}

int Antecedent::get_length() const {
    int count = 0;
    for (int idx : antecedent_indices) {
        if (idx != 0)
            count++;
    }
    return count;
}

std::string Antecedent::get_linguistic_representation() const {
    std::ostringstream oss;
    bool first = true;
    for (int i = 0; i < get_array_size(); i++) {
        if (antecedent_indices[i] != 0) {
            FuzzyVariable* var = knowledge->get_fuzzy_variable(i);
            std::string term = var->get_fuzzy_set(antecedent_indices[i])->get_term();
            if (!first) oss << " AND ";
            oss << var->get_name() << " IS " << term;
            first = false;
        }
    }
    if (first) {
        return "[don't care]";
    }
    return oss.str();
}

Knowledge* Antecedent::get_knowledge() const {
    return knowledge;
}

void Antecedent::set_knowledge(Knowledge* new_knowledge) {
    knowledge = new_knowledge;
}

Antecedent::operator std::string() const {
    std::ostringstream oss;
    oss << "[";
    for (auto idx : antecedent_indices) {
        oss << idx << " ";
    }
    oss << "]";
    return oss.str();
}

std::string Antecedent::to_string() const {
    return static_cast<std::string>(*this);
}

bool Antecedent::operator==(const Antecedent& other) const {
    if (get_array_size() != other.get_array_size())
        return false;

    for (int i = 0; i < static_cast<int>(antecedent_indices.size()); i++) {
        if (antecedent_indices[i] != other.antecedent_indices[i])
            return false;
    }
    return (*knowledge) == *(other.knowledge);
}

Antecedent* Antecedent::clone() const {
    return new Antecedent(*this);
}
