#include "dont_care_fuzzy_set.hpp"
#include "../membership_function/dont_care_mf.hpp"
#include <memory>

#include "division_type.hpp"
#include "../../knowledge/knowledge.hpp"

DontCareFuzzySet::DontCareFuzzySet(int id) 
    : FuzzySet(std::make_shared<DontCareMF>(), id, DivisionType::EQUAL_DIVISION, "DC") {
}

DontCareFuzzySet::DontCareFuzzySet(const DontCareFuzzySet& other) 
    : FuzzySet(other) {
}
