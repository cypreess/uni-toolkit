import importlib


def import_path(path):
    """Import class from the given absolute python path"""

    module_path = path.split('.')[:-1]
    class_name = path.split('.')[-1]
    module = importlib.import_module('.'.join(module_path))
    return getattr(module, class_name)
