class RejectedClassLabelException(Exception):
    def __init__(self, message: str = "The class label is rejected"):
        super().__init__(message)
