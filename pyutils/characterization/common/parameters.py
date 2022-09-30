import pandas as pd

from pyutils.common.strings import S_PARAMETERS_GENERATOR_VERSION


class BaseParameters:
    def __init__(self, name="base"):
        self.generator_version = -1
        self.name = name

    def __str__(self):
        return self.to_str()

    def to_dict(self):
        raise NotImplemented

    def to_list(self, exclude: list = ()):
        if len(exclude) == 0:
            return [v for k, v in self.to_dict().items() if k not in (S_PARAMETERS_GENERATOR_VERSION,)]

        if S_PARAMETERS_GENERATOR_VERSION not in exclude:
            exclude.append(S_PARAMETERS_GENERATOR_VERSION)

        return [v for k, v in self.to_dict().items() if k not in exclude]

    def to_str(self):
        str_form = "-".join([self.format_parameter(i) for i in self.to_list()])
        return str_form

    def to_csv(self, delimiter=' '):
        obj = ''
        for c in self.to_id():
            obj += str(c)
            obj += delimiter
        return obj[0:len(obj) - 1]

    def to_id(self):
        return self.to_list()

    def to_df(self):
        return pd.DataFrame(self.to_dict(), index=[0])

    @classmethod
    def from_dict(cls, parameters):
        raise NotImplemented

    @classmethod
    def from_list(cls, parameters):
        raise NotImplemented

    @classmethod
    def from_str(cls, parameters):
        raise NotImplemented

    @staticmethod
    def is_equal(parameters_1: dict, parameters_2: dict):
        for key_1 in parameters_1.keys():
            if key_1 == S_PARAMETERS_GENERATOR_VERSION:
                continue
            if key_1 not in parameters_2.keys() or parameters_1[key_1] != parameters_2[key_1]:
                return False
        return True

    @staticmethod
    def format_parameter(param):
        if param is None:
            return "None"
        if isinstance(param, (tuple, list)):
            return "_".join([str(i) for i in param])
        elif isinstance(param, bool):
            res = '1' if param else '0'
            return res
        else:
            return str(param)
