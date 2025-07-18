#include "pattern.hpp"
#include <stdexcept>

Pattern::Pattern(int id, const std::vector<double>& attributes_vector, std::shared_ptr<AbstractClassLabel> target_class)
    : id(id), attributes_vector(attributes_vector), target_class(target_class) {
    if (id < 0) {
        throw std::invalid_argument("ID cannot be negative");
    }
    if (attributes_vector.empty()) {
        throw std::invalid_argument("Attributes vector can't be empty");
    }
    if (!target_class) {
        throw std::invalid_argument("Target class can't be null");
    }
}

Pattern::Pattern(const Pattern& other)
    : id(other.id), attributes_vector(other.attributes_vector), target_class(other.target_class) {
}

int Pattern::get_id() const {
    return id;
}

const std::vector<double>& Pattern::get_attributes_vector() const {
    return attributes_vector;
}

double Pattern::get_attribute_value(int index) const {
    if (index < 0 || index >= static_cast<int>(attributes_vector.size())) {
        throw std::out_of_range("Index out of bounds");
    }
    return attributes_vector[index];
}

std::shared_ptr<AbstractClassLabel> Pattern::get_target_class() const {
    return target_class;
}

int Pattern::get_num_dim() const {
    return attributes_vector.size();
}

bool Pattern::operator==(const Pattern& other) const
{
    return id == other.id &&
           target_class == other.target_class &&
           attributes_vector == other.attributes_vector;
}

Pattern::operator std::string() const
{
    if (attributes_vector.empty() || !target_class) {
        return "null";
    }
    std::string txt = "[id:" + std::to_string(id) + ", input:{";
    for (size_t i = 0; i < attributes_vector.size(); ++i) {
        txt += std::to_string(attributes_vector[i]);
        if (i < attributes_vector.size() - 1) {
            txt += ", ";
        }
    }
    txt += "}, Class:" + std::string(*target_class) + "]";
    return txt;
}

Pattern* Pattern::clone() const
{
    return new Pattern(*this);
}

std::string Pattern::to_string() const {
    return static_cast<std::string>(*this);
}
