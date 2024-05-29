from abc import ABC, abstractmethod


class AbstractClassLabel:
    _class_label = None
    __rejected_class_label = -1

    def __init__(self, class_label):
        self._class_label = class_label

    def get_class_label_value(self):
        if self._class_label is None:
            raise Exception('Class label not defined')
        return self._class_label

    def set_class_label_value(self, class_label):
        self._class_label = class_label

    def get_rejected_class_label_value(self):
        return self.__rejected_class_label
