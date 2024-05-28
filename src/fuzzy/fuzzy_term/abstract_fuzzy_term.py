from abc import ABC, abstractmethod


class AbstractFuzzyTerm(ABC):
    # __partition_type = None
    # __partition_num = None
    # __partition_id = None

    # def __init__(self, partition_type, partition_num, partition_id):
        # self.__partition_type = partition_type
        # self.__partition_num = partition_num
        # self.__partition_id = partition_id
        # pass

    # def get_partition_type(self):
    #     return self.__partition_type
    #
    # def get_partition_num(self):
    #     return self.__partition_num
    #
    # def get_partition_id(self):
    #     return self.__partition_id

    @abstractmethod
    def get_membership_value(self, value):
        pass
