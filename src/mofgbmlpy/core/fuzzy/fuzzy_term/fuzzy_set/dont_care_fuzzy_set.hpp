#ifndef DONT_CARE_FUZZY_SET_HPP
#define DONT_CARE_FUZZY_SET_HPP

#include "fuzzy_set.hpp"
#include <memory>

class DontCareFuzzySet : public FuzzySet {
public:
    DontCareFuzzySet(int id);
    DontCareFuzzySet(const DontCareFuzzySet& other);
    virtual ~DontCareFuzzySet() = default;
};

#endif // DONT_CARE_FUZZY_SET_HPP
