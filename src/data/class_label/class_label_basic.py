from src.data.class_label.abstract_class_label import AbstractClassLabel


class ClassLabelBasic(AbstractClassLabel):
    def __init__(self, class_label):
        super().__init__(class_label)

    def __eq__(self, other):
        if not isinstance(other, ClassLabelBasic):
            return False
        return other.get_class_label_value() == self.get_class_label_value()

    def __copy__(self):
        return ClassLabelBasic(self.get_class_label_value())

    def __str__(self):
        if self.get_class_label_value() is None:
            raise Exception("class label value is None")
        return f"{self.get_class_label_value():2d}"

