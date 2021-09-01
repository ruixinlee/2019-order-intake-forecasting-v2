# custom_exception.py
# defines all exception


class LuigiTaskFailure(Exception):
    """
    Luigi Exception Catcher on task failure
    """
    def __init__(self):
        pass


class ModelRunError(LuigiTaskFailure):
    """
    Raised when model fails to run
    """
    def __init__(self):
        super(ModelRunError, self).__init__()