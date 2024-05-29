from enum import Enum


class ExperienceParameter:
    class ClassLabelType(Enum):
        SINGLE = 0
        MULTI = 1

    class DivisionType(Enum):
        EQUAL_DIVISION = 0
        ENTROPY_DIVISION = 0

    class ShapeTypeName(Enum):
        GAUSSIAN = 0
        TRAPEZOID = 1
        INTERVAL = 2
        TRIANGLE = 3

    class ObjectivesForMichigan(Enum):
        FITNESS_VALUE = 0
        RULE_LENGTH = 1

    class ObjectivesForPittsburgh(Enum):
        ERROR_RATE_DTRA = 0
        NUMBER_OF_RULE = 1
        ERROR_RATE_DTST = 2

    __instance = None
    _class_label_type = ClassLabelType.SINGLE

    def get_class_label_type(self):
        return self._class_label_type

    def set_class_label_type(self, new_value):
        if new_value < 0 or new_value > 1:
            raise ValueError("Value must be between 0 and 1")
        self._class_label_type = new_value

    def __new__(cls, *args, **kwargs):
        if ExperienceParameter.__instance is None:
            ExperienceParameter.__instance = super(ExperienceParameter, cls).__new__(cls)
        return ExperienceParameter.__instance

    @staticmethod
    def get_instance():
        if ExperienceParameter.__instance is None:
            ExperienceParameter.__new__(ExperienceParameter)
        return ExperienceParameter.__instance
