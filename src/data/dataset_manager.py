#TODO: DELETE THIS FILE

# class DataSetManager:
#     __instance = None
#
#     __trains = []
#     __tests = []
#
#     def __new__(cls, *args, **kwargs):
#         if DataSetManager.__instance is None:
#             DataSetManager.__instance = super(DataSetManager, cls).__new__(cls)
#         return DataSetManager.__instance
#
#     @staticmethod
#     def get_instance():
#         if DataSetManager.__instance is None:
#             DataSetManager.__new__(DataSetManager)
#         return DataSetManager.__instance
#
#     def add_train(self, train):
#         self.__trains.append(train)
#
#     def add_test(self, test):
#         self.__tests.append(test)
#
#     def get_trains(self):
#         if self.__trains is None:
#             raise ValueError("trains hasn't been initialized")
#         return self.__trains
#
#     def get_tests(self):
#         if self.__tests is None:
#             raise ValueError("tests hasn't been initialized")
#         return self.__tests
#
#     def clear(self):
#         self.__trains = []
#         self.__tests = []