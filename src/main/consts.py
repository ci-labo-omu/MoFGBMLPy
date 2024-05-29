class Consts:
    # Experimental Settings
    POPULATION_SIZE = 60
    OFFSPRING_POPULATION_SIZE = 60
    TERMINATE_GENERATION = 5000
    TERMINATE_EVALUATION = 300000
    OUTPUT_FREQUENCY = 6000

    # Random Number Seed
    RAND_SEED = 2020

    # OS
    WINDOWS = 0     # Windows
    UNIX = 1        # Mac or Linux

    # Fuzzy Classifier
    IS_DONT_CARE_PROBABILITY = False
    ANTECEDENT_NUMBER_NOT_DONT_CARE = 5
    DONT_CARE_RT = 0.8
    INITIATION_RULE_NUM = 30
    MAX_RULE_NUM = 60
    MIN_RULE_NUM = 1

    # FGBML
    MICHIGAN_OPE_RT = 0.5
    RULE_CHANGE_RT = 0.2
    MICHIGAN_CROSS_RT = 0.9
    PITTSBURGH_CROSS_RT = 0.9
    FUZZY_GRADE = 1.0

    # Folders' Name
    ROOTFOLDER = "results"
    ALGORITHM_ID_DIR = "ALGORITHM_ID"
    EXPERIMENT_ID_DIR = "EXPERIMENT_ID"

    # Index
    TRAIN = 0
    TEST = 1
    XML_FILE_NAME = "results_XML"

    # Dataset info
    DATA_SIZE = 0
    ATTRIBUTE_NUMBER = 0
    CLASS_LABEL_NUMBER = 0

    def set(self, src):
        # TODO
        pass

    def __str__(self):
        # TODO
        return "(not implemented)"
