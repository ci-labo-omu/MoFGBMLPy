#include "triangular_fuzzy_set.hpp"
#include <memory>

TriangularFuzzySet::TriangularFuzzySet(float left, float center, float right, int id, const std::string& term) 
    : FuzzySet(new TriangularMF(left, center, right), id, DivisionType::EQUAL_DIVISION, term) {
}

TriangularFuzzySet::TriangularFuzzySet(const TriangularFuzzySet& other) 
    : FuzzySet(other) {
}
