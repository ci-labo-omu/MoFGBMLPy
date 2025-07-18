#ifndef RECTANGULAR_FUZZY_SET_HPP
#define RECTANGULAR_FUZZY_SET_HPP

#include "fuzzy_set.hpp"
#include <memory>
#include <string>

class RectangularFuzzySet : public FuzzySet {
public:
    RectangularFuzzySet(float left, float right, int id, const std::string& term);
    RectangularFuzzySet(const RectangularFuzzySet& other);
    virtual ~RectangularFuzzySet() = default;
};

#endif // RECTANGULAR_FUZZY_SET_HPP
