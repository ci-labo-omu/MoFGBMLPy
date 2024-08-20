class AbstractMethodException(Exception):
    def __init__(self, message: str = "This method is abstract. Please use a concrete class."):
        super().__init__(message)
