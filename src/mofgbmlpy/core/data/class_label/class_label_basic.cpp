#include "class_label_basic.hpp"

ClassLabelBasic::ClassLabelBasic(int class_label) : AbstractClassLabel(), class_label(class_label) {
}

ClassLabelBasic::ClassLabelBasic(const ClassLabelBasic& other) 
    : AbstractClassLabel(other), class_label(other.class_label) {
}

int ClassLabelBasic::get_class_label_value() const {
    return class_label;
}

void ClassLabelBasic::set_class_label_value(int new_val) {
    class_label = new_val;
}

ClassLabelBasic::operator std::string() const
{
    return std::to_string(class_label);
}

bool ClassLabelBasic::operator==(const AbstractClassLabel& other) const
{
    const ClassLabelBasic* other_basic = dynamic_cast<const ClassLabelBasic*>(&other);
    if (!other_basic) {
        return false;
    }
    return class_label == other_basic->class_label;
}

ClassLabelBasic* ClassLabelBasic::clone() const
{
    return new ClassLabelBasic(*this);
}

