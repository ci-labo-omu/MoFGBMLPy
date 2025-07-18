#include "rectangular_fuzzy_set.hpp"
#include "../membership_function/rectangular_mf.hpp"
#include <memory>

#include "division_type.hpp"

RectangularFuzzySet::RectangularFuzzySet(float left, float right, int id, const std::string& term) 
    : FuzzySet(std::make_shared<RectangularMF>(left, right), id, DivisionType::EQUAL_DIVISION, term) {
}

RectangularFuzzySet::RectangularFuzzySet(const RectangularFuzzySet& other) 
    : FuzzySet(other) {
}
