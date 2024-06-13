from abc import ABC


class Arguments(ABC):
    __values = {}

    def __init__(self):
        # TODO: Check if those parameters are all useful

        # Experimental Settings
        self.__values["POPULATION_SIZE"] = 60
        self.__values["OFFSPRING_POPULATION_SIZE"] = 60
        self.__values["TERMINATE_GENERATION"] = 5000
        self.__values["TERMINATE_EVALUATION"] = 300000
        self.__values["OUTPUT_FREQUENCY"] = 6000

        # Random Number Seed
        self.__values["RAND_SEED"] = 2020

        # OS
        self.__values["WINDOWS"] = 0     # Windows
        self.__values["UNIX"] = 1        # Mac or Linux

        # Fuzzy Classifier
        self.__values["IS_DONT_CARE_PROBABILITY"] = False
        self.__values["ANTECEDENT_NUM_NOT_DONT_CARE"] = 5
        self.__values["DONT_CARE_RT"] = 0.8
        self.__values["INITIATION_RULE_NUM"] = 30
        self.__values["MAX_NUM_RULES"] = 60
        self.__values["MIN_NUM_RULES"] = 1

        # FGBML
        self.__values["MICHIGAN_OPE_RT"] = 0.5
        self.__values["RULE_CHANGE_RT"] = 0.2
        self.__values["MICHIGAN_CROSS_RT"] = 0.9
        self.__values["PITTSBURGH_CROSS_RT"] = 0.9
        self.__values["FUZZY_GRADE"] = 1.0

        # Folders' Name
        self.__values["ROOT_FOLDER"] = "results"
        self.__values["ALGORITHM_ID_DIR"] = "ALGORITHM_ID"
        self.__values["EXPERIMENT_ID_DIR"] = "EXPERIMENT_ID"

        # Index
        self.__values["TRAIN"] = 0
        self.__values["TEST"] = 1
        self.__values["XML_FILE_NAME"] = "results_XML"

        # Dataset info
        self.__values["DATA_SIZE"] = 0
        self.__values["ATTRIBUTE_NUMBER"] = 0
        self.__values["CLASS_LABEL_NUMBER"] = 0

    def set(self, key, value):
        self.__values[str(key)] = value

    def get(self, key):
        return self.__values[key]

    def get_keys(self):
        return list(self.__values.keys())

    def has_key(self, key):
        return key in self.__values

    def __str__(self):
        txt = ""
        for key, value in self.__values.items():
            txt += f"{key} = {value}\n"

        return txt
