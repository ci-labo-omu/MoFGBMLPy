from abc import ABC, abstractmethod


class AbstractClassLabel:
    _class_label = None
    __is_rejected = False

    def __init__(self, class_label):
        self._class_label = class_label

    def get_class_label_value(self):
        if self._class_label is None:
            raise Exception('Class label not defined')
        return self._class_label

    def set_class_label_value(self, class_label):
        self._class_label = class_label

    def is_rejected(self):
        return self.__is_rejected

    def set_rejected(self):
        self.__is_rejected = True
