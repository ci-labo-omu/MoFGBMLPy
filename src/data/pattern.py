class Pattern:
    __id = None
    __attribute_vector = None
    __target_class = None

    def __init__(self, pattern_id, __attribute_vector, target_class):
        if pattern_id < 0:
            raise ValueError('id must be positive')
        elif __attribute_vector is None:
            raise ValueError('attribute_vector must not be None')
        elif target_class is None:
            raise ValueError('target_class must not be None')

        self.__id = pattern_id
        self.__attribute_vector = __attribute_vector
        self.__target_class = target_class

    def get_id(self):
        return self.__id

    def get_attributes_vector(self):
        return self.__attribute_vector

    def get_attribute_value(self, index):
        return self.__attribute_vector[index]

    def get_target_class(self):
        return self.__target_class

    def __str__(self):
        if self.get_attributes_vector() is None or self.get_target_class() is None:
            return "null"

        return f"[id:{self.get_id()}, input:{{{self.get_attributes_vector()}}}, Class:{self.get_target_class()}]"
