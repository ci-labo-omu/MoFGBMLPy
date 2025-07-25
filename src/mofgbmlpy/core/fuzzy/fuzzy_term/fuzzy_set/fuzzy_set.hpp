#ifndef FUZZY_SET_HPP
#define FUZZY_SET_HPP

#include <string>
#include <memory>

#include "division_type.hpp"
#include "../membership_function/abstract_mf.hpp"  // your AbstractMF class should be polymorphic and cloneable

class FuzzySet {
private:
    AbstractMF* function;
    std::string term;
    int id;
    DivisionType division_type;

public:
    FuzzySet(AbstractMF* function, int id, DivisionType division_type, std::string term = "");
    FuzzySet(const FuzzySet& other);
    operator std::string() const;
    std::string to_string() const;

    AbstractMF* get_function() const;
    void set_function(AbstractMF* new_function);
    std::string get_term() const;
    int get_id() const;
    int get_division_type() const;

    float get_membership_value(float x) const;
    float get_support(float x_min = 0.0f, float x_max = 0.0f) const;

    bool operator==(const FuzzySet& other) const;
    FuzzySet* clone() const;
};

#endif
