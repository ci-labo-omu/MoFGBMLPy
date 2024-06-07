#TODO: DELETE THIS FILE

# import numpy as np
#
#
# class AttributeVector:
#     __attribute_vector = None
#
#     def __init__(self, attribute_vector):
#         if attribute_vector is None:
#             raise ValueError("Attribute vector cannot be None")
#         self.__attribute_vector = np.copy(attribute_vector)
#
#     def get_attribute_array(self):
#         if self.__attribute_vector is None:
#             raise ValueError("Attribute vector has not been initialized")
#         return self.__attribute_vector
#
#     def get_value_at(self, index):
#         if len(self.__attribute_vector) <= index or index < 0:
#             raise IndexError("Index out of bounds")
#         return self.__attribute_vector[index]
#
#     def get_num_dim(self):
#         return len(self.__attribute_vector)
#
#     def __str__(self):
#         if self.__attribute_vector is None or len(self.__attribute_vector) == 0:
#             return "null"
#         else:
#             txt = f"{self.__attribute_vector[0]:.4f}"
#             for i in range(1, len(self.__attribute_vector)):
#                 txt += f", {self.__attribute_vector[i]:.4f}"
#             return txt