import glob
import os
from os.path import join as join_path

import pandas as pd
import xgboost as xgb

from pyutils.analysis.common.data_transformers import Transformers
from pyutils.characterization.kernels.utils.checks import KernelsChecks
from pyutils.characterization.networks.utils import NetworksNameHelpers, get_unified_benchmark_name
from pyutils.common.config import ALL_LABELS
from pyutils.common.methods import is_power, is_runtime
from pyutils.common.paths import SAVED_MODELS_DIR
from pyutils.common.strings import S_CONV, S_BATCHNORM, S_MATMUL, S_RELU, S_MAXPOOL, \
    S_ADAPTIVEPOOL, S_ADD, S_CAT
from pyutils.common.strings import S_RUNTIME_MS, S_AVG_PWR_W
from pyutils.common.utils import GlobalLogger, TimeStamp, HDMY_DATE_FORMAT
from pyutils.modeling.kernels.kernels_training_models import ConvModel, BatchNormModel, MatMulModel, MaxPoolModel, \
    AdaptivePoolModel, ReluModel, CatModel, AddModel
from pyutils.modeling.networks.networks_training_models import NetworksBaseModel

logger = GlobalLogger().get_logger()

S_NW_INFERENCE_TIME = 'Inference Time'
S_NW_INFERENCE_POWER = 'Inference Power'
S_NW_INFERENCE_ENERGY = 'Inference Energy'
S_NW_INFERENCE_ENERGY_PREDICTION_ERR = "Inference Energy Prediction Relative Error %"
S_NW_INFERENCE_RUNTIME_PREDICTION_ERR = "Inference Runtime Prediction Relative Error %"
S_NW_INFERENCE_POWER_PREDICTION_ERR = "Inference Power Prediction Relative Error %"


class PredictionHelpers:
    @staticmethod
    def is_energy_data(field):
        field = str(field).lower()
        if field.__contains__('energy'):
            return True

    @staticmethod
    def is_runtime_data(field):
        field = str(field).lower()
        if field.__contains__('time') or field.__contains__('runtime'):
            return True


class TrainingHelpers:
    @staticmethod
    def load_kernel_model(kernel_name):
        if KernelsChecks.is_conv2d(kernel_name):
            model = ConvModel()
        elif KernelsChecks.is_bn(kernel_name):
            model = BatchNormModel()
        elif KernelsChecks.is_mm(kernel_name):
            model = MatMulModel()
        elif KernelsChecks.is_maxpool(kernel_name):
            model = MaxPoolModel()
        elif KernelsChecks.is_adaptivepool(kernel_name):
            model = AdaptivePoolModel()
        elif KernelsChecks.is_relu(kernel_name):
            model = ReluModel()
        elif KernelsChecks.is_cat(kernel_name):
            model = CatModel()
        elif KernelsChecks.is_add(kernel_name):
            model = AddModel()
        else:
            return None
        return model

    @staticmethod
    def load_network_model(network_name):
        nw_model = NetworksBaseModel()
        nw_model.network = get_unified_benchmark_name(network_name)
        return nw_model

    @staticmethod
    def get_label(metric):
        if is_power(metric):
            return S_AVG_PWR_W
        elif is_runtime(metric):
            return S_RUNTIME_MS
        else:
            raise Exception(f"Unknown metric. {metric}")


class TrainedModelWrapper:
    def __init__(self, model: xgb.Booster):
        self.model = model

    def predict(self, data: pd.DataFrame):
        data.drop(columns=[i for i in data.columns if i in ALL_LABELS], inplace=True)
        prediction_input_dm = xgb.DMatrix(data)
        prediction = self.model.predict(prediction_input_dm)
        return Transformers.inverse_log_transform(prediction)


class ModelSelectionFilter:
    def __init__(self, on=None, before=None, after=None, n_from_last=None):
        self.on = on
        self.before = before
        self.after = after
        self.n_from_last = n_from_last

    def get_criteria(self):
        if self.on is not None:
            return 'on'
        elif self.n_from_last is not None:
            return 'n_from_last'
        elif self.before is not None and self.after is not None:
            return 'range'
        elif self.before is not None:
            return 'before'
        elif self.after is not None:
            return 'after'
        else:
            return 'latest'


class ModelLoader:
    @staticmethod
    def get_trained_model_dir(target, label, selection_filter: ModelSelectionFilter = None,
                              parent_dir=SAVED_MODELS_DIR):
        if selection_filter is None:
            selection_filter = ModelSelectionFilter()

        saved_models_dir = join_path(parent_dir, target)
        if not os.path.exists(saved_models_dir):
            logger.error(f'Saved models directory {saved_models_dir} does not exist. Exiting.')
            return

        # List all folders (i.e., timestamp)
        models_parent_folders_paths = glob.glob(f'{saved_models_dir}/**/*[0-9]')
        if len(models_parent_folders_paths) == 0:
            # Check sub-directory
            models_parent_folders_paths = glob.glob(f'{saved_models_dir}/*[0-9]')
            if len(models_parent_folders_paths) == 0:
                raise Exception(f"No saved models for: {target}. Exiting.")

        # Manage labels: select only folders that contain the label model
        filtered_models_parent_folders_paths = list()
        for model_parent_path in models_parent_folders_paths:
            sub_files = os.listdir(model_parent_path)
            for sub_file in sub_files:
                if sub_file.__contains__(label):
                    filtered_models_parent_folders_paths.append(model_parent_path)
                    break

        # Check that there are folders in the filtered list
        if len(filtered_models_parent_folders_paths) == 0:
            raise Exception(f"Filtered saved models for: {target} based on {label} is empty. Exiting.")

        # Parse all timestamps and create a dictionary to map timestamps to paths
        timestamp_path_map = dict()
        timestamps = list()

        for i in filtered_models_parent_folders_paths:
            timestamp_str = os.path.basename(i)
            timestamp_path_map[timestamp_str] = i
            timestamps.append(TimeStamp.parse_timestamp(timestamp_str))

        # Sort in ascending order
        timestamps.sort()

        if selection_filter.get_criteria() == 'latest':
            target_timestamp = timestamps[-1]
        elif selection_filter.get_criteria() == 'n_from_last':
            idx = int(-1 * int(selection_filter.n_from_last))
            target_timestamp = timestamps[idx]
        elif selection_filter.get_criteria() == 'on':
            target_timestamp = None
            for i in timestamps:
                if i == TimeStamp.parse_timestamp(selection_filter.on):
                    target_timestamp = i
                    break
            if target_timestamp is None:
                raise Exception(f"Not found timestamp on: {selection_filter.on}")
        elif selection_filter.get_criteria() == 'range':
            raise Exception("Not implemented")
        elif selection_filter.get_criteria() == 'before':
            # Assume that we want the last timestamp before this before timestamp
            # TODO: fix the logic here to account for different sizes
            before_timestamp = TimeStamp.parse_timestamp(selection_filter.before)
            target_timestamp = None
            for i in range(len(timestamps)):
                if timestamps[i] == before_timestamp:
                    target_timestamp = timestamps[i - 1]
                    break
            if target_timestamp is None:
                raise Exception(f"Not found timestamp before: {selection_filter.before}")
        elif selection_filter.get_criteria() == 'after':
            # Assume that we want the last timestamp after this timestamp
            # TODO: fix the logic here to account for different sizes
            after_timestamp = TimeStamp.parse_timestamp(selection_filter.after)
            target_timestamp = None
            for i in range(len(timestamps)):
                if timestamps[i] == after_timestamp:
                    target_timestamp = timestamps[i + 1]
                    break
            if target_timestamp is None:
                raise Exception(f"Not found timestamp after: {selection_filter.after}")
        else:
            raise Exception(f"Unknown selection criteria. {selection_filter.get_criteria()}")

        alternative_value = TimeStamp.to_str(target_timestamp, date_format=HDMY_DATE_FORMAT)
        model_parent_folder_path = timestamp_path_map.get(TimeStamp.to_str(target_timestamp),
                                                          timestamp_path_map.get(alternative_value))
        return model_parent_folder_path

    @staticmethod
    def get_trained_model_path(kernel, label, selection_filter=None):
        latest_model_parent_path = ModelLoader.get_trained_model_dir(kernel, label, selection_filter)
        latest_timestamp_str = os.path.basename(latest_model_parent_path)

        # Load the latest model file
        model_file_basename = f'xgb_{label}_{latest_timestamp_str}.model'
        model_path = join_path(latest_model_parent_path, model_file_basename)
        return model_path

    @staticmethod
    def get_trained_model_meta_path(kernel, label, selection_filter=None, parent_dir=SAVED_MODELS_DIR):
        latest_model_parent_path = ModelLoader.get_trained_model_dir(kernel, label, selection_filter,
                                                                     parent_dir=parent_dir)
        latest_timestamp_str = os.path.basename(latest_model_parent_path)

        # Load the latest model file
        model_meta_file_basename = f'xgb_meta_{label}_{latest_timestamp_str}.json'
        for file_name in os.listdir(latest_model_parent_path):
            if file_name.__contains__('meta') and file_name.__contains__(label):
                model_meta_file_basename = file_name

        model_path = join_path(latest_model_parent_path, model_meta_file_basename)
        return model_path

    @staticmethod
    def load_trained_model(kernel, label, selection_filter=None):
        """ This method loads the latest trained model for a specific kernel and label (e.g., Time, power). Default
        behavior is to load the latest model, otherwise it searches for the model using the specified timestamp. """
        model_path = ModelLoader.get_trained_model_path(kernel, label, selection_filter)
        # Create model object and load it
        model = xgb.Booster()
        logger.info(f"Loading latest model file for {kernel}:{label} at {model_path}")
        model.load_model(model_path)
        return model

    @staticmethod
    def load_all_kernels_models(metric: str, container: dict = None,
                                selection_filter: ModelSelectionFilter = None) -> dict:
        if container is None:
            prediction_models = dict()
        else:
            prediction_models = container
        for kernel in KernelsChecks.get_supported_kernels():
            prediction_models[kernel] = TrainedModelWrapper(ModelLoader.load_trained_model(kernel, metric,
                                                                                           selection_filter))
        return prediction_models

    @staticmethod
    def load_all_networks_models(metric: str, container: dict = None,
                                 selection_filter: ModelSelectionFilter = None) -> dict:
        if container is None:
            prediction_models = dict()
        else:
            prediction_models = container
        for network in NetworksNameHelpers.get_supported_networks():
            prediction_models[network] = TrainedModelWrapper(ModelLoader.load_trained_model(network, metric,
                                                                                            selection_filter))
        return prediction_models

    @staticmethod
    def load_specific_models(metric, container, models_dict):
        if container is None:
            prediction_models = dict()
        else:
            prediction_models = container
        for kernel in KernelsChecks.get_supported_kernels():
            sf = models_dict.get(kernel)
            prediction_models[kernel] = TrainedModelWrapper(ModelLoader.load_trained_model(kernel, metric, sf))
        return prediction_models

    @staticmethod
    def get_runtime_models_timestamp():
        d = dict()
        d[S_CONV] = '1625042021'
        d[S_BATCHNORM] = '1704022021'
        d[S_MATMUL] = '2005022021'
        d[S_RELU] = '2103022021'
        d[S_MAXPOOL] = '1409022021'
        d[S_ADAPTIVEPOOL] = '1208022021'
        d[S_ADD] = '1516022021'
        d[S_CAT] = '1209022021'

        for key in d.keys():
            d[key] = ModelSelectionFilter(on=d.get(key))
        return d

    @staticmethod
    def get_power_models_timestamp():
        d = dict()
        d[S_CONV] = '1419052021'
        d[S_BATCHNORM] = '0118052021'
        d[S_MATMUL] = '1120052021'
        d[S_RELU] = '2317052021'
        d[S_MAXPOOL] = '1718052021'
        d[S_ADAPTIVEPOOL] = '1630042021'
        d[S_ADD] = '1518052021'
        d[S_CAT] = '1118052021'
        for key in d.keys():
            d[key] = ModelSelectionFilter(on=d.get(key))
        return d


if __name__ == '__main__':
    pass
