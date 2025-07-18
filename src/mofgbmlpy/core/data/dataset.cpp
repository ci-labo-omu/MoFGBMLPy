#include "dataset.hpp"
#include <stdexcept>

Dataset::Dataset(int size, int num_dim, int num_classes, const std::vector<Pattern*>& patterns)
    : size(size), num_dim(num_dim), num_classes(num_classes), patterns(patterns) {
    if (size <= 0 || num_dim <= 0 || num_classes <= 0) {
        throw std::invalid_argument("size, num_dim and num_classes must be positive");
    }
    if (patterns.empty()) {
        throw std::invalid_argument("Patterns vector can't be empty");
    }
    if (size != static_cast<int>(patterns.size())) {
        throw std::invalid_argument("Size doesn't match patterns vector size");
    }
    Pattern first_pattern = *patterns[0];
    if (num_dim != first_pattern.get_num_dim()) {
        throw std::invalid_argument("Number of dimensions does not match the first pattern's dimensions");
    }
}

Dataset::Dataset(const Dataset& other)
    : size(other.size), num_dim(other.num_dim), num_classes(other.num_classes), patterns(other.patterns) {
}

Pattern* Dataset::get_pattern(int index) const {
    if (index < 0 || index >= size) {
        throw std::out_of_range("Index out of bounds");
    }
    return patterns[index];
}

const std::vector<Pattern*>& Dataset::get_patterns() const {
    return patterns;
}

int Dataset::get_num_dim() const {
    return num_dim;
}

int Dataset::get_num_classes() const {
    return num_classes;
}

int Dataset::get_size() const {
    return size;
}

Dataset::operator std::string() const
{
    if (patterns.empty()) {
        return "null";
    }
    std::string txt = std::to_string(size) + ", " + std::to_string(num_dim) + ", " + std::to_string(num_classes) + "\n";
    for (const auto& pattern : patterns) {
        txt += std::string(*pattern) + "\n";
    }
    return txt;
}

bool Dataset::operator==(const Dataset& other) const
{
    return size == other.size &&
           num_dim == other.num_dim &&
           num_classes == other.num_classes &&
           patterns == other.patterns;
}

Dataset* Dataset::clone() const
{
    return new Dataset(*this);
}

std::string Dataset::to_string() const {
    return static_cast<std::string>(*this);
}
