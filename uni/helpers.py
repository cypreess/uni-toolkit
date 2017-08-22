import importlib


def import_path(path):
    """Import class from the given absolute python path"""

    module_path = path.split('.')[:-1]
    class_name = path.split('.')[-1]
    module = importlib.import_module('.'.join(module_path))
    return getattr(module, class_name)


def parse_boolean(value):
    if type(value) is not bool:
        return not (value.strip().lower() in ('', 'no', 'false', '0'))
    return value


def type_or_none(t):
    def _(value):
        if not value:
            return None
        return t(value)
    return _
