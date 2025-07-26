#include "dont_care_fuzzy_set.hpp"

DontCareFuzzySet::DontCareFuzzySet(int id) 
    : FuzzySet(new DontCareMF(), id, DivisionType::EQUAL_DIVISION, "DC") {
}

DontCareFuzzySet::DontCareFuzzySet(const DontCareFuzzySet& other) 
    : FuzzySet(other) {
}
