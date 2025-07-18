#include "abstract_mf.hpp"
#include <algorithm>

AbstractMF::AbstractMF(const std::vector<float>& params, bool are_params_points_flag)
    : params(params), are_params_points_flag(are_params_points_flag) {}

std::vector<float> AbstractMF::get_params() const {
    return params;
}

bool AbstractMF::is_param_value_valid(int index, float value, float x_min, float x_max) const {
    std::vector<float> param_range = get_param_range(index, x_min, x_max);
    if (param_range.empty()) {
        return false; // No valid range for this parameter
    }
    return param_range[0] <= value && value <= param_range[1];
}

bool AbstractMF::are_params_points() const {
    return are_params_points_flag;
}

void AbstractMF::set_param_value(int index, float value, float x_min, float x_max) {
    if (!is_param_value_valid(index, value, x_min, x_max)) {
        throw std::invalid_argument("Parameter value is not valid");
    }
    if (index < 0 || index >= static_cast<int>(params.size())) {
        throw std::out_of_range("Parameter index out of range");
    }
    params[index] = value;
}

bool AbstractMF::operator==(const AbstractMF& other) const {
    return params == other.params;
}

AbstractMF::operator std::string() const {
    return "Abstract membership function";
}

std::string AbstractMF::to_string() const {
    return static_cast<std::string>(*this);
}
