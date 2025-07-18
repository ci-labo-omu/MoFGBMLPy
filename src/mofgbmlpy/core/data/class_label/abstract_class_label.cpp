#include "abstract_class_label.hpp"

AbstractClassLabel::AbstractClassLabel() : is_rejected_flag(false) {
}

bool AbstractClassLabel::is_rejected() const {
    return is_rejected_flag;
}

void AbstractClassLabel::set_rejected() {
    is_rejected_flag = true;
}

std::string AbstractClassLabel::to_string() const
{
    return static_cast<std::string>(*this);
}
