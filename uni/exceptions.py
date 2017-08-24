class UniConfigurationError(Exception):
    def __init__(self, message):
        self.message = message


class UniFatalError(Exception):
    def __init__(self, message):
        self.message = message