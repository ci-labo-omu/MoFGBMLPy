class ExceededMaxTrialsNumber(Exception):
    def __init__(self, message: str = "Exceeded the maximum number of trials"):
        super().__init__(message)
