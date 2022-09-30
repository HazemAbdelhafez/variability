import logging
import os
import os.path as path
import pathlib
from datetime import datetime


class GlobalLogger:
    # Logger initialization

    def __init__(self, _logging_level=logging.INFO):
        if _logging_level == logging.DEBUG:
            logging_format = '%(asctime)s [%(levelname)s] File %(filename)s Line No. %(lineno)d  %(message)s'
        else:
            logging_format = '%(asctime)s [%(levelname)s] %(message)s'

        logging.basicConfig(format=logging_format,
                            datefmt="%Y-%m-%d-%H-%M-%S")
        self._logger = logging.getLogger('main')
        self._logger.setLevel(_logging_level)

    def get_logger(self):
        return self._logger


logger = GlobalLogger().get_logger()

PROGRESS_DIR_PATH = "/local/hazem/projects/nodes-status"

NUM_NODES = 14


def get_all_files(p):
    return [path.join(os.path.abspath(p), i) for i in os.listdir(p)]


def get_modification_timestamp(p):
    if type(p) is str:
        p = get_path_obj(p)
    return p.stat().st_mtime


def get_file_type(p):
    if p.__contains__('.'):
        return p.split('.')[-1]
    else:
        return None


def get_path_obj(p):
    return pathlib.Path(p)


def get_latest_timestamp(f):
    with open(f) as f_obj:
        line = f_obj.readlines()[-1]
        line = line.rstrip('\n')
        line = line.split(']')[0].lstrip('[')
        d = datetime.strptime(line, "%d/%m/%Y %H:%M:%S")
        return d


def get_node_files(node_id, all_files: list):
    started_files = list()
    done_file = ''
    error_count = 0
    warning_count = 0
    for f in all_files:
        node_of_file = os.path.basename(f).split('_')[0]
        if node_id == node_of_file:
            if f.__contains__('done'):
                done_file = f
            elif f.__contains__('started'):
                started_files.append(f)
            elif f.__contains__('warning'):
                warning_count += 1
            else:
                logger.warning("File does not contain either started or done or warning keywords: %s" % f)
                error_count += 1

    return started_files, done_file, error_count, warning_count


def check_node_progress(node_id, all_files):
    started_files, done_file, error_count, warning_count = get_node_files(node_id, all_files)
    if done_file == '' or error_count != 0:
        # logger.info("Not done yet.")
        return False

    # Check that the done file is correct: its timestamp is post all the started files timestamps.
    # 1.1 Get all time stamps
    content_timestamps = list()
    modification_timestamps = list()
    for f in started_files:
        content_timestamps.append(get_latest_timestamp(f))
        modification_timestamps.append(get_modification_timestamp(f))

    # 1.2 Sort them
    content_timestamps.sort(reverse=True)
    modification_timestamps.sort(reverse=True)

    # 1.3 pick the latest
    latest_content_timestamp = content_timestamps[0]
    latest_modification_timestamp = modification_timestamps[0]

    done_latest_content_timestamp = get_latest_timestamp(done_file)
    done_latest_modification_timestamp = get_modification_timestamp(done_file)

    diff = latest_modification_timestamp - done_latest_modification_timestamp
    if diff > 0:
        # logger.warning("Done timestamp should be after started timestamp.")
        return True

    diff = (latest_content_timestamp - done_latest_content_timestamp).total_seconds()
    if diff > 0:
        # logger.error("Done timestamp should be after started timestamp. NOT DONE.")
        return False

    return True


def main():
    files = get_all_files(PROGRESS_DIR_PATH)
    logger.info("Checking progress on %d nodes." % NUM_NODES)

    # Check number of files
    n = len(files)
    if n < 2 * NUM_NODES:
        logger.warning("Only %d progress files so far. Check later. NOT DONE." % n)
        return -1
    logger.info("Found %d progress files." % n)

    # # Check that we have half n done and half n started
    # started_files = list()
    # done_files = list()
    # for f in files:
    #     if f.__contains__('done'):
    #         done_files.append(f)
    #     elif f.__contains__('started'):
    #         started_files.append(f)
    #     else:
    #         logger.warning("File does not contain either started or done keywords: %s" % f)
    #
    # s = len(started_files)
    # d = len(done_files)
    # if s != d:
    #     logger.error("Done and started files count must match. NOT DONE.")
    #     return -1
    # logger.info("Found %d started and done files." % d)
    #
    # started_files.sort()
    # done_files.sort()
    #
    # # Read files contents and extract latest timestamps
    # for sf, df in zip(started_files, done_files):
    #     st = get_modification_timestamp(get_path_obj(sf))
    #     dt = get_modification_timestamp(get_path_obj(df))
    #     diff = st - dt
    #     if diff > 0:
    #         logger.warning("Done timestamp should be after started timestamp.")
    #         return -1
    #
    #     st = get_latest_timestamp(sf)
    #     dt = get_latest_timestamp(df)
    #     diff = (dt-st).total_seconds()
    #     if diff < 0:
    #         logger.error("Done timestamp should be after started timestamp. NOT DONE.")
    #         return -1
    #
    # logger.info("Job done.")

    nodes_done = list()
    nodes_in_progress = list()
    for i in range(1, NUM_NODES + 1):
        node_id = 'node-%02d' % i
        node_status = check_node_progress(node_id, files)
        if node_status:
            nodes_done.append(node_id)
        else:
            nodes_in_progress.append(node_id)

    logger.info("%d Nodes done: %s " % (len(nodes_done), str(nodes_done)))
    if len(nodes_in_progress) != 0:
        logger.info("%d Nodes still in progress: %s " % (len(nodes_in_progress), str(nodes_in_progress)))
        return -1
    return 0


if __name__ == '__main__':
    import sys

    ret_value = main()
    if ret_value != 0:
        sys.exit(1)
    else:
        sys.exit(0)
