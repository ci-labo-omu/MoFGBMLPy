#include "class_label_multi.hpp"
#include <stdexcept>

ClassLabelMulti::ClassLabelMulti(const std::vector<int>& class_label) 
    : AbstractClassLabel(), class_label(class_label) {
}

ClassLabelMulti::ClassLabelMulti(const ClassLabelMulti& other) 
    : AbstractClassLabel(other), class_label(other.class_label) {
}


int ClassLabelMulti::get_length() const
{
    return class_label.size();
}

const std::vector<int>& ClassLabelMulti::get_class_label_value() const {
    return class_label;
}

int ClassLabelMulti::get_class_label_value_at(int index) const
{
    return class_label.at(index);
}

void ClassLabelMulti::set_class_label_vector(const std::vector<int>& class_label)
{
    this->class_label = class_label;
}

ClassLabelMulti::operator std::string() const
{
    std::string txt = std::to_string(class_label[0]);
    if (class_label.size() > 1) {
        for (size_t i = 1; i < class_label.size(); ++i) {
            txt += ", " + std::to_string(class_label[i]);
        }
    }
    return txt;
}

bool ClassLabelMulti::operator==(const AbstractClassLabel& other) const
{
    const ClassLabelMulti* other_multi = dynamic_cast<const ClassLabelMulti*>(&other);
    if (!other_multi) {
        return false;
    }
    return class_label == other_multi->class_label;
}

ClassLabelMulti* ClassLabelMulti::clone() const
{
    return new ClassLabelMulti(*this);
}

void ClassLabelMulti::set_class_label_value(const std::vector<int>& new_value) {
    class_label = new_value;
}

