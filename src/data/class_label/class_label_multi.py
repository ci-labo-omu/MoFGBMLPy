from data.class_label.abstract_class_label import AbstractClassLabel


class ClassLabelMulti(AbstractClassLabel):
    def __init__(self, class_label):
        super().__init__(class_label)

    def __eq__(self, other):
        if not isinstance(other, ClassLabelMulti) or self.get_length() != other.get_length():
            return False

        for i in range(self.get_length()):
            if self.get_class_label_value()[i] != other.get_class_label_value()[i]:
                return False
        return True

    def get_length(self):
        return len(self.get_class_label_value())

    def __copy__(self):
        return ClassLabelMulti(self.get_class_label_value())

    def __str__(self):
        if self.get_class_label_value() is None:
            raise Exception("class label value is None")
        txt = f"{self.get_class_label_value()[0]:2d}"

        if self.get_length() > 1:
            for i in range(1, self.get_length()):
                txt = f"{txt}, {self.get_class_label_value()[i]:2d}"

        return txt

