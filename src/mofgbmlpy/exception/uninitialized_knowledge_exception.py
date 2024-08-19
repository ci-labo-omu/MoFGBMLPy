class UninitializedKnowledgeException(Exception):
    def __init__(self, message: str = "Knowledge is not yet initialized (no fuzzy set)"):
        super().__init__(message)
