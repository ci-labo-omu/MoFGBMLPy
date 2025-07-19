#ifndef CLASS_LABEL_MULTI_HPP
#define CLASS_LABEL_MULTI_HPP

#include "abstract_class_label.hpp"
#include <vector>

class ClassLabelMulti : public AbstractClassLabel {
private:
    std::vector<int> class_label;

public:
    ClassLabelMulti(const std::vector<int>& class_label);
    ClassLabelMulti(const ClassLabelMulti& other);
    virtual ~ClassLabelMulti() = default;
    
    int get_length() const;
    void set_class_label_value(const std::vector<int>& class_label);
    const std::vector<int>& get_class_label_value() const;
    int get_class_label_value_at(int index) const;

    operator std::string() const override;
    bool operator==(const AbstractClassLabel& other) const override;
    ClassLabelMulti* clone() const override;
};

#endif // CLASS_LABEL_MULTI_HPP
