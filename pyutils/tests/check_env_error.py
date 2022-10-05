import os, psutil
from time import sleep

from pyutils.characterization.driver import characterize_model_time
from pyutils.common.experiments_utils import logger
from pyutils.common.utils import FileUtils
import pyutils.hosts.agx as agx


def get_env_data():
    pid = os.getpid()
    p = psutil.Process(pid)
    env_vars = p.environ()
    mem_mapping = p.memory_maps(grouped=False)
    return {"pid": pid, "environment_variables": env_vars, "memory_maps": mem_mapping}


def main():
    counter = 0
    while True:
        t = get_env_data()
        FileUtils.serialize(user_data=t, file_path="/tmp/tmp.json", append=True)
        logger.info(f"I am done at {counter}")
        counter += 1
        sleep(0.5)


def main2():
    config = FileUtils.deserialize("/tmp/characterization_job_cfg.json")
    while True:
        characterize_model_time(config)


if __name__ == '__main__':
    # agx.BoardController.reset_dvfs_settings()
    main2()
