class Context:
    __instance = None

    __dont_care_id = 0
    __fuzzy_sets = []

    def __new__(cls, *args, **kwargs):
        if Context.__instance is None:
            Context.__instance = super(Context, cls).__new__(cls)
        return Context.__instance

    @staticmethod
    def get_instance():
        if Context.__instance is None:
            Context.__new__(Context)
        return Context.__instance

    def get_fuzzy_set(self, dim, fuzzy_set_id=None):
        if self.__fuzzy_sets is None or len(self.__fuzzy_sets) == 0:
            raise Exception("Context is not yet initialized (no fuzzy set)")

        if fuzzy_set_id is None:
            return self.__fuzzy_sets[dim]
        else:
            return self.__fuzzy_sets[dim][fuzzy_set_id]

    def get_fuzzy_set_num(self, dim):
        if self.__fuzzy_sets is None or len(self.__fuzzy_sets) == 0:
            raise Exception("Context is not yet initialized (no fuzzy set)")

        return len(self.__fuzzy_sets[dim])

    def set_fuzzy_sets(self, fuzzy_sets):
        if self.__fuzzy_sets is not None and len(self.__fuzzy_sets) != 0:
            raise Exception("You can't overwrite fuzzy sets. You must call clear before doing so")

        self.__fuzzy_sets = fuzzy_sets

    def get_fuzzy_sets(self):
        if self.__fuzzy_sets is None or len(self.__fuzzy_sets) == 0:
            raise Exception("Context is not yet initialized (no fuzzy set)")

        return self.__fuzzy_sets

    def get_membership_value(self, attribute_value, dim, fuzzy_set_id):
        if self.__fuzzy_sets is None or len(self.__fuzzy_sets) == 0:
            raise Exception("Context is not yet initialized (no fuzzy set)")
        return self.__fuzzy_sets[dim][fuzzy_set_id].get_membership_value(attribute_value)

    def get_num_dim(self):
        if self.__fuzzy_sets is None:
            return 0
        return len(self.__fuzzy_sets)

    def clear(self):
        self.__fuzzy_sets = []

    def __str__(self):
        txt = ""
        for i in range(self.get_num_dim()):
            for item in self.__fuzzy_sets[i]:
                txt = f"{txt}{str(item)}\n"
        return txt
