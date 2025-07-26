#ifndef DONT_CARE_FUZZY_SET_HPP
#define DONT_CARE_FUZZY_SET_HPP

#include "fuzzy_set.hpp"
#include <memory>
#include "division_type.hpp"
#include "../../knowledge/knowledge.hpp"
#include "../membership_function/dont_care_mf.hpp"

class DontCareFuzzySet : public FuzzySet {
public:
    DontCareFuzzySet(int id);
    DontCareFuzzySet(const DontCareFuzzySet& other);
    virtual ~DontCareFuzzySet() = default;
};

#endif // DONT_CARE_FUZZY_SET_HPP
