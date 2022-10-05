import datetime
import errno
import inspect
import json
import logging
import os
import pathlib
import pickle
from json.decoder import JSONDecodeError
from os.path import join
from typing import List

import pandas as pd

from pyutils.common.methods import join
from pyutils.common.paths import PROJECT_DIR, CACHE_DIR, FIGURES_DIR

from pyutils.common.methods import json_convert

HDMY_DATE_FORMAT = '%H%d%m%Y'
MHDMY_DATE_FORMAT = '%M%H%d%m%Y'
READABLE_MHDMY_DATE_FORMAT = '%Y_%m_%d-%H_%M'
READABLE_HDMY_DATE_FORMAT = '%Y_%m_%d-%H'


class FileUtils:
    @staticmethod
    def serialize(user_data, file_path=None, file_extension=None, save_as='object',
                  append=False, prefix_timestamp=None, indent=None):
        if not append:
            FileUtils.silent_remove(file_path)
        if file_path is None:
            print(f"No output file specified. "
                  f"Writing to tmp_save_{str(datetime.datetime.now().strftime(HDMY_DATE_FORMAT))}.{file_extension}")

        parent_dir = os.path.dirname(file_path)
        if parent_dir != "" and not os.path.exists(parent_dir):
            print(f"[WARNING] Parent directory {parent_dir} did not exist. Creating it before serializing")
            os.makedirs(parent_dir)

        if file_extension is None:
            inferred_extension = str(file_path).split('.')
            if len(inferred_extension) == 0:
                file_extension = '.txt'
            else:
                if inferred_extension[1] in ['json', 'txt', 'text', 'dat', 'pickle', 'parquet', 'pkl']:
                    file_extension = inferred_extension[1]

        file_extension = file_extension.strip('.')

        if not str(file_path).__contains__(f'.{file_extension}'):
            file_path += f'.{file_extension}'

        if file_extension == 'pickle' or file_extension == 'dat' or file_extension == 'pkl':
            file_mode = 'ab+' if append else 'wb+'
            with open(file_path, file_mode) as file_obj:
                pickle.dump(user_data, file_obj)

        elif file_extension == 'json':
            file_mode = 'a+' if append else 'w+'
            with open(file_path, file_mode) as file_obj:
                if isinstance(user_data, List) and save_as == 'list':
                    for item in user_data:
                        json.dump(item, file_obj, default=json_convert, indent=indent)
                        file_obj.write("\n")
                else:
                    json.dump(user_data, file_obj, default=json_convert, indent=indent)
                    if append:
                        file_obj.write("\n")

        elif file_extension == 'txt' or file_extension == 'text':
            file_mode = 'a+' if append else 'w+'
            with open(file_path, file_mode) as file_obj:
                if isinstance(user_data, List):
                    for line in user_data:
                        if line == '\n':
                            continue
                        if prefix_timestamp is not None:
                            file_obj.write(f'[{prefix_timestamp}] {line}')
                        else:
                            file_obj.write(str(line))
                        file_obj.write("\n")
                else:
                    if prefix_timestamp is not None:
                        file_obj.write(f'[{prefix_timestamp}] {user_data}')
                    else:
                        file_obj.write(str(user_data))
                    if append:
                        file_obj.write("\n")

        elif file_extension == 'parquet':
            if not isinstance(user_data, pd.DataFrame):
                user_data = pd.DataFrame(user_data)
            user_data.to_parquet(file_path)
        return file_path

    @staticmethod
    def deserialize(file_path, file_extension=None, return_type='object'):
        if not os.path.exists(file_path):
            raise FileNotFoundError(str(file_path))

        if file_extension is None:
            file_extension = str(file_path.split('.')[-1])

        if file_extension == 'pickle' or file_extension == 'dat' or file_extension == 'pkl':
            with open(file_path, 'rb') as file_obj:
                return pickle.load(file_obj)
        elif file_extension == 'json':
            try:
                with open(file_path, 'r') as file_obj:
                    data = json.load(file_obj)
                    if return_type == 'list' and type(data) != list:
                        return [data]
                    return data
            except JSONDecodeError as _:
                data = list()
                for line in open(file_path, 'r'):
                    if line == '\n':
                        continue
                    else:
                        try:
                            data.append(json.loads(line))
                        except JSONDecodeError as _:
                            print(f"Ignoring malformed JSON line: {line}")
                return data
        elif file_extension == 'txt' or return_type == 'raw':
            lines = []
            with open(file_path, 'r') as file_obj:
                for line in file_obj.readlines():
                    line = line.replace("\n", '')
                    lines.append(line)
            return lines
        elif file_extension == 'csv':
            return pd.read_csv(file_path)

        elif file_extension == 'parquet':
            return pd.read_parquet(file_path)

        else:
            raise Exception(f"File extension not supported {file_extension}")

    @staticmethod
    def is_json(file_name: str):
        if file_name.__contains__('json'):
            return True

    @staticmethod
    def silent_remove(file_path):
        try:
            os.remove(file_path)
        except OSError as e:
            if e.errno != errno.ENOENT:  # errno.ENOENT = no such file or directory
                raise

    @staticmethod
    def exists(file_path):
        return os.path.exists(file_path)

    @staticmethod
    def to_csv(df: pd.DataFrame, output_file_path):
        parent_dir = os.path.dirname(output_file_path)
        if parent_dir != "" and not os.path.exists(parent_dir):
            print(f"[WARNING] Parent directory {parent_dir} did not exist. Creating it before serializing")
            os.makedirs(parent_dir)
        FileUtils.silent_remove(output_file_path)
        df.to_csv(output_file_path, index=False)

    @staticmethod
    def to_pickle(df: pd.DataFrame, output_file_path):
        parent_dir = os.path.dirname(output_file_path)
        if parent_dir != "" and not os.path.exists(parent_dir):
            print(f"[WARNING] Parent directory {parent_dir} did not exist. Creating it before serializing")
            os.makedirs(parent_dir)
        FileUtils.silent_remove(output_file_path)
        df.to_pickle(output_file_path)


def prepare(*args):
    """ A method that takes input file path tree and prepares it by creating missing directories."""
    if len(args) == 0:
        return ''
    elif len(args) == 1:
        p = pathlib.Path(args[0])
    else:
        p = pathlib.Path(args[0])
        for i in args[1:]:
            if isinstance(i, datetime.datetime):
                p /= pathlib.Path(str(i.year))
                p /= pathlib.Path(str(i.month))
                p /= pathlib.Path(str(i.day))
            else:
                p /= pathlib.Path(i)

    # Maybe not a very good assumption: every file has an extension, otherwise, it is a dir.
    is_file = str(p).__contains__(".")

    if is_file:
        if not p.parent.exists():
            logging.getLogger('FileUtils').warning(f"Creating non existing directory: {p.parent}")
            os.makedirs(p.parent, exist_ok=True)
    else:
        # Make sure the parent directory exists if file, and that the directory exists if dir
        if not p.exists():
            logging.getLogger('FileUtils').warning(f"Creating non existing directory: {p}")
            os.makedirs(p, exist_ok=True)
    return str(p)


class TimeStamp:
    @staticmethod
    def get_timestamp(date_format: str = HDMY_DATE_FORMAT):
        return datetime.datetime.now().strftime(date_format)

    @staticmethod
    def get_minute_timestamp():
        return datetime.datetime.now().strftime('%Y_%m_%d-%H_%M')

    @staticmethod
    def get_hour_timestamp():
        return datetime.datetime.now().strftime('%Y_%m_%d-%H')

    @staticmethod
    def parse_timestamp(timestamp: str):
        for possible_format in [READABLE_MHDMY_DATE_FORMAT, READABLE_HDMY_DATE_FORMAT, MHDMY_DATE_FORMAT,
                                HDMY_DATE_FORMAT]:
            try:
                date_obj = datetime.datetime.strptime(timestamp, possible_format)
                return date_obj
            except ValueError:
                continue
        # To avoid raising an error over timestamp issue, we resort to this unified date until we figure it out.
        return datetime.datetime.strptime("2222_22_22-22_22", READABLE_MHDMY_DATE_FORMAT)

    @staticmethod
    def to_str(timestamp, date_format: str = READABLE_MHDMY_DATE_FORMAT):
        return timestamp.strftime(date_format)


class GlobalLogger:
    # Logger initialization

    def __init__(self, _logging_level=logging.INFO):
        if _logging_level == logging.DEBUG:
            logging_format = '%(asctime)s [%(levelname)s] File %(filename)s Line No. %(lineno)d  %(message)s'
        else:
            logging_format = '%(asctime)s [%(levelname)s] %(message)s'

        logging.basicConfig(format=logging_format, datefmt="%Y-%m-%d-%H-%M-%S")
        self._logger = logging.getLogger('main')
        self._logger.setLevel(_logging_level)

    def get_logger(self):
        return self._logger

    def set_out_file(self, p):
        p = prepare(p)
        file_handler = logging.FileHandler(p, mode='w')
        self._logger.addHandler(file_handler)


def get_pymodule_dir_path(parent_dir, file_name: str):
    _file_name = file_name.split('/')[-1].rstrip(".py")
    p = join(parent_dir, *__name__.split('.')[1:-1], _file_name)
    return prepare(p)


class DiskCache:
    def __init__(self, identity: str = None, clear=False):
        self.clear = clear

        current_frame = inspect.currentframe()
        call_frames = inspect.getouterframes(current_frame, 3)

        # This will return the method that called create_cache
        calling_method_name = str(call_frames[1].frame.f_back.f_code.co_name)

        calling_file_name = str(call_frames[2].filename.rstrip(".py"))
        calling_file_relative_path = calling_file_name.replace(os.path.join(PROJECT_DIR, "pyutils/"), "")
        assert os.path.exists(join(PROJECT_DIR, "pyutils", f"{calling_file_relative_path}.py"))
        cache_dir_path = prepare(CACHE_DIR, calling_file_relative_path, calling_method_name)
        if identity is not None and identity != '':
            self.cache_file_path = join(cache_dir_path, f"{identity}.pickle")
        else:
            self.cache_file_path = join(cache_dir_path, f"default.pickle")

    def valid(self):
        return os.path.exists(self.cache_file_path) and not self.clear

    def save(self, data):
        FileUtils.serialize(data, file_path=self.cache_file_path, file_extension="pickle", append=False)
        return data

    def load(self):
        if self.valid():
            return FileUtils.deserialize(self.cache_file_path)
        else:
            return None


def create_cache(*args, clear=False):
    def _preprocess(id_element):
        # TODO: update this method as necessary.
        id_element = str(id_element).lower().replace(".json", "")
        return id_element

    identity = '_'.join([_preprocess(i) for i in args])
    return DiskCache(identity, clear)


if __name__ == '__main__':
    pass


def get_figures_dir_path(file_name: str):
    return get_pymodule_dir_path(FIGURES_DIR, file_name)


def get_tmp_dir_path(file_name: str):
    return get_pymodule_dir_path("/tmp", file_name)