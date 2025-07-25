#include "triangular_fuzzy_set.hpp"
#include "../membership_function/triangular_mf.hpp"
#include <memory>

#include "division_type.hpp"

TriangularFuzzySet::TriangularFuzzySet(float left, float center, float right, int id, const std::string& term) 
    : FuzzySet(new TriangularMF(left, center, right), id, DivisionType::EQUAL_DIVISION, term) {
}

TriangularFuzzySet::TriangularFuzzySet(const TriangularFuzzySet& other) 
    : FuzzySet(other) {
}
