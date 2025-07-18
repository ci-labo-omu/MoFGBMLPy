#ifndef TRIANGULAR_FUZZY_SET_HPP
#define TRIANGULAR_FUZZY_SET_HPP

#include "fuzzy_set.hpp"
#include <string>

class TriangularFuzzySet : public FuzzySet {
public:
    TriangularFuzzySet(float left, float center, float right, int id, const std::string& term);
    TriangularFuzzySet(const TriangularFuzzySet& other);
    virtual ~TriangularFuzzySet() = default;
};

#endif // TRIANGULAR_FUZZY_SET_HPP
