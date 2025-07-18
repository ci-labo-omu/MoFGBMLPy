#include "dont_care_mf.hpp"
#include <stdexcept>

DontCareMF::DontCareMF() : AbstractMF({}) {
}

float DontCareMF::get_value(float x) const {
    return 1.0f;
}

DontCareMF::operator std::string() const {
    return "<Dont Care MF>";
}

std::vector<float> DontCareMF::get_param_range(int index, float x_min, float x_max) const {
    return {};
}

std::vector<std::vector<float>> DontCareMF::get_plot_points(float x_min, float x_max) const {
    std::vector<std::vector<float>> points = {
        { x_min, 1.0f },
        { x_max, 1.0f }
    };
    return points;
}

float DontCareMF::get_support(float x_min, float x_max) const {
    return x_max - x_min;
}

DontCareMF* DontCareMF::clone() const {
    return new DontCareMF();
}

