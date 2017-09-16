import importlib
import inspect
import json
import os
from pathlib import Path

from uni.exceptions import UniFatalError


def import_path(path):
    """Import class from the given absolute python path"""

    module_path = path.split('.')[:-1]
    class_name = path.split('.')[-1]
    module = importlib.import_module('.'.join(module_path))
    return getattr(module, class_name)


def str_choices(choices):
    def _(value):
        value = str(value)
        assert value in choices, "Value %s not in %s" % (value, choices)
        return value

    return _


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


class ParameterReaderMixin:
    def read_parameters(self):
        """
        Reading default parameters from provided file
        """
        base_dir = os.path.dirname(inspect.getfile(self.__class__))
        parameters_file = os.path.join(base_dir, 'parameters.json')
        if Path(parameters_file).is_file():
            try:
                with open(parameters_file) as f:
                    return json.load(f)
            except json.decoder.JSONDecodeError as e:
                raise UniFatalError("Cannot parse parameters file (%s): %s" % (parameters_file, e))
        return {}
