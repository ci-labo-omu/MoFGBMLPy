class Pattern:
    __id = None
    __attribute_vector = None
    __target_class = None

    def __init__(self, id, attribute_vector, target_class):
        if id < 0:
            raise ValueError('id must be positive')
        elif attribute_vector is None:
            raise ValueError('attribute_vector must not be None')
        elif target_class is None:
            raise ValueError('target_class must not be None')

        self.__id = id
        self.__attribute_vector = attribute_vector
        self.__target_class = target_class

    def get_id(self):
        return self.__id

    def get_attribute_vector(self):
        if self.__attribute_vector is None:
            raise ValueError('attribute_vector is not initialized yet')
        return self.__attribute_vector

    def get_attribute_array(self):
        if self.__attribute_vector is None:
            raise ValueError('attribute_vector is not initialized yet')
        return self.__attribute_vector.get_attribute_array()

    def get_attribute_value_at(self, index):
        return self.__attribute_vector.get_attribute_value_at(index)

    def get_target_class(self):
        return self.__target_class

    def __str__(self):
        if self.get_attribute_vector() is None or self.get_target_class() is None:
            return "null"

        return f"[id:{self.get_id()}, input:{{{self.get_attribute_vector()}}}, Class:{self.get_target_class()}]"