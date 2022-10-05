import os
import subprocess as sp
from os.path import join as join_path

from pyutils.common.strings import S_CORE_AFFINITY, S_NUM_ISOLATED_CORES, S_NUM_THREADS, S_DISABLE_ASLR
from pyutils.common.utils import GlobalLogger, FileUtils, TimeStamp
from pyutils.hosts.agx import AGX_CPU_CORE_COUNT
from pyutils.hosts.common import HOSTNAME

logger = GlobalLogger().get_logger()

ALPS_NODES = ['node-%02d' % i for i in range(1, 15)]


class PathsHandler:
    @staticmethod
    def get_project_root_dir():
        if HOSTNAME == 'xps':
            return "/home/ubuntu/projects/variability"
        elif HOSTNAME == 'node-15':
            return "/ssd/projects/variability"
        elif HOSTNAME in ALPS_NODES:
            return "/home/ubuntu/variability"
        elif HOSTNAME in ['node260', 'node240']:
            return "/local/hazem/projects/variability"
        else:
            return None

    @staticmethod
    def get_local_status_dir():
        if HOSTNAME == 'node-15':
            p = "/ssd/nodes-status"
        elif HOSTNAME in ALPS_NODES:
            p = "/home/ubuntu/nodes-status"
        else:
            p = "/tmp/nodes-status"

        if not os.path.exists(p):
            os.makedirs(p, exist_ok=True)
        return p

    @staticmethod
    def get_remote_status_dir():
        if HOSTNAME == 'node-15':
            p = '/ssd/remote-nodes-status'
            if not os.path.exists(p):
                os.makedirs(p, exist_ok=True)
        else:
            p = "/local/hazem/projects/nodes-status"
        return p

    @staticmethod
    def get_python_remote_path():
        # if HOSTNAME == 'node-15':
        #     return "/ssd/miniforge3/envs/cdev/bin/python"
        return "/home/ubuntu/miniconda3/envs/mirage/bin/python"

    @staticmethod
    def get_local_logging_dir():
        p = join_path(PathsHandler.get_project_root_dir(), 'logs', HOSTNAME)
        if not os.path.exists(p):
            os.makedirs(p, exist_ok=True)
        return p


class HostsHandler:
    @staticmethod
    def get_local_hostname():
        return HOSTNAME

    @staticmethod
    def get_remote_logger_hostname():
        if HOSTNAME == 'node-15':
            return 'node-15'
        return "node260"

    @staticmethod
    def get_remote_logger_username():
        if HOSTNAME == 'node-15':
            return 'ubuntu'
        return "hazem"


class EnvHandler:
    @staticmethod
    def set_env(job_config: dict):
        if job_config[S_CORE_AFFINITY]:
            starting_core_id = AGX_CPU_CORE_COUNT - int(job_config[S_NUM_ISOLATED_CORES])
            # A necessary change to run the inference threads on all isolated cores. We use FIFO.
            pid = str(os.getpid())
            os.system(f"taskset -cap {starting_core_id}-{AGX_CPU_CORE_COUNT - 1} {pid}")
            os.system(f"chrt -fap 1 {pid}")
            os.environ['OMP_PROC_BIND'] = 'close'
            os.environ['OMP_PLACES'] = '{' + ','.join([str(i) for i in range(starting_core_id,
                                                                             AGX_CPU_CORE_COUNT)]) + '}'
            os.environ['OMP_NUM_THREADS'] = str(job_config[S_NUM_THREADS])

        # TODO: re-enable this
        # if job_config[S_DISABLE_ASLR]:
        #     # Read via: /proc/sys/kernel/randomize_va_space
        #     os.system("sysctl -w kernel.randomize_va_space=0")

        # Set python path to avoid module not found exceptions.
        os.environ['PYTHONPATH'] = f"{os.getenv('PYTHONPATH')}:{PathsHandler.get_project_root_dir()}/pyutils"


class Logging:
    default_remote_hostname = HostsHandler.get_remote_logger_hostname()
    default_remote_status_dir = PathsHandler.get_remote_status_dir()
    default_remote_username = HostsHandler.get_remote_logger_username()

    @staticmethod
    def write_and_send(msg, file_path, host=default_remote_hostname, dir_path=default_remote_status_dir,
                       file_ext='txt', append=True,
                       timestamp_format='%d/%m/%Y %H:%M:%S'):
        Logging.write_locally(msg, file_path, file_ext, append, timestamp_format)
        Logging.send(file_path, logger_hostname=host, logger_dest_path=dir_path)

    @staticmethod
    def write_locally(msg, file_path, file_ext='txt', append=True, timestamp_format='%d/%m/%Y %H:%M:%S'):
        FileUtils.serialize(msg, file_path=file_path, file_extension=file_ext,
                            prefix_timestamp=TimeStamp.get_timestamp(timestamp_format), append=append)

    @staticmethod
    def send(file_path, logger_hostname=default_remote_hostname, logger_dest_path=default_remote_status_dir,
             logger_username=default_remote_username):
        Logging.send_progress_report(file_path, host=logger_hostname, dir_path=logger_dest_path,
                                     logger_username=logger_username)

    @staticmethod
    def send_progress_report(local_status_file, host, dir_path, local_username='ubuntu',
                             logger_username="ubuntu"):
        logger.info(f"Sending progress report to {host}...")
        os.system(f"scp -i /home/{local_username}/.ssh/id_rsa -o StrictHostKeyChecking=no"
                  f" {local_status_file} {logger_username}@{host}"
                  f":{dir_path}/")
        logger.info("Done sending report.")
