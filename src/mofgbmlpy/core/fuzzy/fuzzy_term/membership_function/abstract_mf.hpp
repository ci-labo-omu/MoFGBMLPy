#pragma once
#include <vector>
#include <stdexcept>

class AbstractMF {
protected:
    std::vector<float> params;
    bool are_params_points_flag;

public:
    AbstractMF(const std::vector<float>& params, bool are_params_points_flag=true);
    virtual ~AbstractMF() = default;

    virtual float get_value(float x) const = 0;
    std::vector<float> get_params() const;
    virtual std::vector<float> get_param_range(int index, float x_min=0.f, float x_max=1.f) const = 0;
    bool are_params_points() const;
    bool is_param_value_valid(int index, float value, float x_min=0.f, float x_max=1.f) const;
    void set_param_value(int index, float value, float x_min=0.f, float x_max=1.f);
    virtual std::vector<std::vector<float>> get_plot_points(float x_min=0.f, float x_max=1.f) const = 0;
    virtual float get_support(float x_min=0.f, float x_max=1.f) const = 0;
    bool operator==(const AbstractMF& other) const;
    virtual operator std::string() const;
    std::string to_string() const;
    virtual AbstractMF* clone() const = 0;
};
