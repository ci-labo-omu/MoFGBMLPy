#ifndef RECTANGULAR_FUZZY_SET_HPP
#define RECTANGULAR_FUZZY_SET_HPP

#include "fuzzy_set.hpp"
#include <memory>
#include <string>
#include "../membership_function/rectangular_mf.hpp"
#include "division_type.hpp"

class RectangularFuzzySet : public FuzzySet {
public:
    RectangularFuzzySet(float left, float right, int id, const std::string& term);
    RectangularFuzzySet(const RectangularFuzzySet& other);
    virtual ~RectangularFuzzySet() = default;
};

#endif // RECTANGULAR_FUZZY_SET_HPP
