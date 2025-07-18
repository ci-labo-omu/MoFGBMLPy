#include "fuzzy_set.hpp"

#include <utility>

FuzzySet::FuzzySet(std::shared_ptr<AbstractMF> function, int id, DivisionType division_type, std::string term)
    : function(std::move(function)), term(std::move(term)), id(id), division_type(division_type) {
    if (function == nullptr) {
        throw std::invalid_argument("Function cannot be null");
    }
    // TODO division_type is not yet used
}

FuzzySet::FuzzySet(const FuzzySet& other)
    : function(other.function->clone()), term(other.term), id(other.id), division_type(other.division_type) {}

FuzzySet::operator std::string() const {
    return "FuzzySet " + term;
}

std::string FuzzySet::to_string() const {
    return static_cast<std::string>(*this);
}

float FuzzySet::get_membership_value(float x) const {
    return function->get_value(x);
}

std::string FuzzySet::get_term() const {
    return term;
}

int FuzzySet::get_id() const {
    return id;
}

std::shared_ptr<AbstractMF> FuzzySet::get_function() const {
    return function;
}

void FuzzySet::set_function(std::shared_ptr<AbstractMF> function) {
    this->function = std::move(function);
}

int FuzzySet::get_division_type() const {
    return division_type;
}

float FuzzySet::get_support(float x_min, float x_max) const {
    return function->get_support(x_min, x_max);
}

bool FuzzySet::operator==(const FuzzySet& other) const {
    return id == other.id &&
           *function == *(other.function) &&
           term == other.term &&
           division_type == other.division_type;
}

FuzzySet* FuzzySet::clone() const {
    return new FuzzySet(*this);
}
