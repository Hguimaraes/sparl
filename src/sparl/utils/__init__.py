from sparl.utils.white_noise import ControlledNoise
from sparl.utils.emphasis import PreEmphasis
from sparl.utils.emphasis import DeEmphasis

__all__ = ['ControlledNoise', 'ConfigObject', 'PreEmphasis', 'DeEmphasis']

class ConfigObject:
    """Transform a dictionary in a object"""
    def __init__(self, **entries):
        self.__dict__.update(entries)