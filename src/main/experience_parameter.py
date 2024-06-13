from enum import IntEnum


class ExperienceParameter:
    # TODO check class usage
    class ClassLabelType(IntEnum):
        SINGLE = 0
        MULTI = 1

    class DivisionType(IntEnum):
        EQUAL_DIVISION = 0
        ENTROPY_DIVISION = 0

    class ShapeTypeName(IntEnum):
        GAUSSIAN = 0
        TRAPEZOID = 1
        INTERVAL = 2
        TRIANGLE = 3

    class ObjectivesForMichigan(IntEnum):
        FITNESS_VALUE = 0
        RULE_LENGTH = 1

    class ObjectivesForPittsburgh(IntEnum):
        ERROR_RATE_DTRA = 0
        NUMBER_OF_RULE = 1
        ERROR_RATE_DTST = 2

    _class_label_type = None

    def get_class_label_type(self):
        return self._class_label_type

    def set_class_label_type(self, new_value):
        int_value = int(new_value)
        if int_value < 0 or int_value > 1:
            raise ValueError("Value must be between 0 and 1")
        self._class_label_type = new_value

    def __init__(self):
        self._class_label_type = ExperienceParameter.ClassLabelType.SINGLE
