#ifndef MOFGBMLPY_CORE_FUZZY_TERM_MEMBERSHIP_FUNCTION_DONT_CARE_MF_HPP
#define MOFGBMLPY_CORE_FUZZY_TERM_MEMBERSHIP_FUNCTION_DONT_CARE_MF_HPP

#include <memory>
#include "abstract_mf.hpp"

class DontCareMF : public AbstractMF {
public:
    DontCareMF();
    
    float get_value(float x) const override;
    std::vector<float> get_param_range(int index, float x_min, float x_max) const override;
    std::vector<std::vector<float>> get_plot_points(float x_min, float x_max) const override;
    float get_support(float x_min, float x_max) const override;
    DontCareMF* clone() const override;
    operator std::string() const override;
};

#endif // MOFGBMLPY_CORE_FUZZY_TERM_MEMBERSHIP_FUNCTION_DONT_CARE_MF_HPP
