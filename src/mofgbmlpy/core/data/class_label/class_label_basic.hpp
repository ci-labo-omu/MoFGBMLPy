#ifndef CLASS_LABEL_BASIC_HPP
#define CLASS_LABEL_BASIC_HPP

#include "abstract_class_label.hpp"

class ClassLabelBasic : public AbstractClassLabel {
private:
    int class_label;

public:
    ClassLabelBasic(int class_label);
    ClassLabelBasic(const ClassLabelBasic& other);
    virtual ~ClassLabelBasic() = default;
    
    int get_class_label_value() const;
    void set_class_label_value(int class_label);

    operator std::string() const override;
    bool operator==(const AbstractClassLabel& other) const override;
    ClassLabelBasic* clone() const override;
};

#endif // CLASS_LABEL_BASIC_HPP
