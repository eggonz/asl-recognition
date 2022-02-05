class InvalidShapeException(Exception):
    """Exception for cases where array shape are incompatible"""
    def __init__(self, *args):
        super(InvalidShapeException, self).__init__(*args)


class NotConfiguredException(Exception):
    """Exception for cases where a certain value is accessed before being configured"""
    def __init__(self, *args):
        super(NotConfiguredException, self).__init__(*args)
