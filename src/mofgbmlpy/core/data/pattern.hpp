#ifndef PATTERN_HPP
#define PATTERN_HPP

#include <vector>
#include <memory>
#include "class_label/abstract_class_label.hpp"

class Pattern {
private:
    int id;
    std::vector<double> attributes_vector;
    std::shared_ptr<AbstractClassLabel> target_class;

public:
    Pattern(int id, const std::vector<double>& attributes_vector, std::shared_ptr<AbstractClassLabel> target_class);
    Pattern(const Pattern& other);
    
    int get_id() const;
    const std::vector<double>& get_attributes_vector() const;
    double get_attribute_value(int index) const;
    std::shared_ptr<AbstractClassLabel> get_target_class() const;
    int get_num_dim() const;

    bool operator==(const Pattern& other) const;
    operator std::string() const;
    std::string to_string() const;
    Pattern* clone() const;
};

#endif // PATTERN_HPP
