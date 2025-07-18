#ifndef ABSTRACT_CLASSIFICATION_HPP
#define ABSTRACT_CLASSIFICATION_HPP

#include <vector>
#include <memory>

#include "../../data/pattern.hpp"
#include "../../gbml/solution/michigan_solution.hpp"

class AbstractClassification {
public:
    AbstractClassification() = default;
    virtual ~AbstractClassification() = default;
    
    virtual std::shared_ptr<MichiganSolution> classify(
        const std::vector<std::shared_ptr<MichiganSolution>>& michigan_solution_list,
        const Pattern& pattern) = 0;
        
    virtual AbstractClassification* clone() const = 0;
    virtual bool operator==(const AbstractClassification& other) const = 0;
    virtual operator std::string() const = 0;
    std::string to_string() const
    {
        return static_cast<std::string>(*this);
    }
};

#endif // ABSTRACT_CLASSIFICATION_HPP
