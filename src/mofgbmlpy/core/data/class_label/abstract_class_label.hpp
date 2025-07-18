#ifndef ABSTRACT_CLASS_LABEL_HPP
#define ABSTRACT_CLASS_LABEL_HPP
#include <string>

class AbstractClassLabel {
private:
    bool is_rejected_flag;

public:
    AbstractClassLabel();
    virtual ~AbstractClassLabel() = default;
    bool is_rejected() const;
    void set_rejected();
    virtual operator std::string() const = 0;
    std::string to_string() const;
    virtual bool operator==(const AbstractClassLabel& other) const = 0;
    virtual AbstractClassLabel* clone() const = 0;
};

#endif // ABSTRACT_CLASS_LABEL_HPP
