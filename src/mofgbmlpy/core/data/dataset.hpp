#ifndef DATASET_HPP
#define DATASET_HPP

#include <vector>
#include <memory>
#include "pattern.hpp"

class Dataset {
private:
    int size;
    int num_dim;  // number of attributes
    int num_classes;
    std::vector<Pattern*> patterns;

public:
    Dataset(int size, int num_dim, int num_classes, const std::vector<Pattern*>& patterns);
    Dataset(const Dataset& other);
    
    Pattern* get_pattern(int index) const;
    const std::vector<Pattern*>& get_patterns() const;
    int get_num_dim() const;
    int get_num_classes() const;
    int get_size() const;

    operator std::string() const;
    std::string to_string() const;
    bool operator==(const Dataset& other) const;
    Dataset* clone() const;
};

#endif // DATASET_HPP
