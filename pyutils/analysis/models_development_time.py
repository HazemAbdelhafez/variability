import numpy as np

from pyutils.characterization.kernels.utils.checks import KernelsChecks
from pyutils.common.strings import S_RUNTIME_MS, S_POWER, S_TRAINING_TIME_SEC
from pyutils.common.utils import FileUtils
from pyutils.common.utils import GlobalLogger
from pyutils.modeling.helpers import ModelLoader

logger = GlobalLogger().get_logger()


def runtime_models_training_time():
    kernels = KernelsChecks.get_supported_kernels()
    training_times = list()
    for kernel in kernels:
        meta_file = ModelLoader.get_trained_model_meta_path(kernel=kernel, label=S_RUNTIME_MS)
        model_info = FileUtils.deserialize(meta_file)
        model_training_time = model_info[S_TRAINING_TIME_SEC]
        training_times.append(model_training_time)

    logger.info("Runtime models development stats: ")
    logger.info(f" -- Min:  {min(training_times) / 3600}")
    logger.info(f" -- Max:  {max(training_times) / 3600}")
    logger.info(f" -- Mean: {np.median(training_times) / 3600}")


def power_models_training_time():
    kernels = KernelsChecks.get_supported_kernels()
    training_times = list()
    for kernel in kernels:
        meta_file = ModelLoader.get_trained_model_meta_path(kernel=kernel, label=S_POWER)
        model_info = FileUtils.deserialize(meta_file)
        model_training_time = model_info[S_TRAINING_TIME_SEC]
        training_times.append(model_training_time)

    logger.info("Power models development stats: ")
    logger.info(f" -- Min:  {min(training_times) / 3600}")
    logger.info(f" -- Max:  {max(training_times) / 3600}")
    logger.info(f" -- Mean: {np.median(training_times) / 3600}")


if __name__ == '__main__':
    runtime_models_training_time()
    power_models_training_time()
