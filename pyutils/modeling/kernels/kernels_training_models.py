import json as js
import os
import traceback
from os.path import join as join_path

import ConfigSpace as cs
import ConfigSpace.hyperparameters as csh
import pandas as pd
import xgboost as xgb
from hpbandster_sklearn import HpBandSterSearchCV
from pyutils.characterization.kernels.adaptivepool_parameters_analysis import AdaptivePool2dNames
from pyutils.characterization.kernels.add_parameters_analysis import AddNames
from pyutils.characterization.kernels.batchnorm_parameters_analysis import BatchNormNames
from pyutils.characterization.kernels.conv2d_parameters_analysis import Conv2dNames
from pyutils.characterization.kernels.matmul_parameters_analysis import MatMulAddNames
from pyutils.characterization.kernels.maxpool2d_parameters_analysis import MaxPool2dNames
from pyutils.characterization.kernels.relu_parameters_analysis import ReluNames
from sklearn.model_selection import LeaveOneGroupOut

from pyutils.analysis.common.data_transformers import Transformers
from pyutils.analysis.common.error_metrics import Metrics
from pyutils.characterization.common.parameters import BaseParameters
from pyutils.common.config import ALL_LABELS, ORIGINAL_LABELS, N_CPU_CORES, HOSTNAME, HOUR_TIMESTAMP
from pyutils.common.data_interface import DataAttributes, DataHandler
from pyutils.common.methods import get_metric
from pyutils.common.paths import SAVED_MODELS_DIR
from pyutils.common.strings import S_NODE_ID, S_RUNTIME_MS, S_RUNTIME, S_DVFS_CONFIG, S_KERNEL_PARAMETERS, \
    S_CHARACTERIZATION, \
    S_MODELING
from pyutils.common.strings import S_PARAMETERS_GENERATOR_VERSION
from pyutils.common.timers import StopWatch
from pyutils.common.utils import FileUtils
from pyutils.common.utils import GlobalLogger
from pyutils.modeling.config import DEFAULT_NUM_BOOST_ROUNDS, EARLY_STOPPING_ROUNDS, LOG_TRANSFORM, \
    DEFAULT_TUNING_ITERATIONS, MIN_BUDGET

logger = GlobalLogger().get_logger()

run_timestamp = HOUR_TIMESTAMP


def xgb_cv(params, _num_boost_rounds, _cv_inner, _groups, _d_train, _x_train, _train_index,
           early_stopping_rounds=EARLY_STOPPING_ROUNDS):
    inner_folds_indexes = []
    for inner_train_index, inner_test_index in _cv_inner.split(_x_train, groups=_groups):
        inner_folds_indexes.append((list(inner_train_index), list(inner_test_index)))
    cv_results = xgb.cv(params, _d_train, num_boost_round=_num_boost_rounds, folds=inner_folds_indexes,
                        nfold=len(inner_folds_indexes), metrics=params['eval_metric'],
                        early_stopping_rounds=early_stopping_rounds, verbose_eval=False, maximize=True,
                        feval=Metrics.XGBoostEval.percentage_error_from_dmatrix)
    metric = 'test-percentage-with-error-less-than-5%-mean'
    mean_percentage_error = cv_results[metric].max()
    best_boost_rounds_idx = cv_results[metric].argmax()
    best_num_boost_rounds = best_boost_rounds_idx + 1  # Arg max return the index which starts from 0
    logger.info("[TRAINING_AND_EVALUATION] Percentage error {} for {} rounds".format(mean_percentage_error,
                                                                                     best_boost_rounds_idx))
    # logger.info(f"[TRAINING_AND_EVALUATION] Validation set best results:  {str(cv_results.iloc[best_boost_rounds_idx])}")
    # logger.info(f"[TRAINING_AND_EVALUATION] End of validation results.")
    return cv_results.iloc[best_boost_rounds_idx], best_num_boost_rounds


class BaseModel:
    def __init__(self):
        self.kernel = "BASE_MODEL_HAS_NO_KERNEL"

    def parse_configurations_from_modeling_data(self, nodes=None):
        return self._parse_configurations_from_experiments_data(mode=S_MODELING, nodes=nodes)

    def parse_configurations_from_characterization_data(self, nodes=None):
        return self._parse_configurations_from_experiments_data(mode=S_CHARACTERIZATION, nodes=nodes)

    def _parse_configurations_from_experiments_data(self, mode=S_MODELING, nodes=None, metric=S_RUNTIME,
                                                    overwrite=False):
        logger.info(f"Parsing configurations for {mode} data for kernel: {self.kernel}")
        if nodes is None:
            raise Exception("Nodes is provided as None.")

        output = list()
        for node in nodes:
            logger.info(f"-- Processing {node}")
            try:
                attr = DataAttributes(node=node, kernel=self.kernel, category=mode, metric=metric, aggregate=True,
                                      overwrite_summary=overwrite)
                records = DataHandler.get_kernel_data(attr)
                for record in records:
                    # TODO: Handle the data cleaning checks (less than time threshold, non empty arrays, etc.),
                    # and metrics
                    configuration = {S_DVFS_CONFIG: record[S_DVFS_CONFIG],
                                     S_KERNEL_PARAMETERS: record[S_KERNEL_PARAMETERS]}
                    output.append(configuration)
            except Exception:
                logger.warning(f"Failed to parse {node}")
        return output

    def parse_and_combine_modeling_data(self, nodes=None, override=False, metric=S_RUNTIME):
        return self._parse_and_combine_experiments_data(mode=S_MODELING, nodes=nodes, override=override,
                                                        metric=metric)

    def parse_and_combine_characterization_data(self, nodes=None, override=False, metric=S_RUNTIME):
        return self._parse_and_combine_experiments_data(mode=S_CHARACTERIZATION, nodes=nodes, override=override,
                                                        metric=metric)

    def _parse_and_combine_experiments_data(self, mode=S_MODELING, nodes=None, override=False, metric=S_RUNTIME):

        logger.info(f"Parsing {mode} data for kernel: {self.kernel}")
        for node in nodes:
            logger.info(f"-- Processing {node} - Metric: {metric}")
            attr = DataAttributes(node=node, kernel=self.kernel, category=mode, metric=metric,
                                  overwrite_summary=override)
            try:
                DataHandler.get_kernel_data(attr)
            except Exception as e:
                st_trace = traceback.format_exc()
                logger.error(e)
                logger.error(st_trace)
                logger.error(f"Parsing data from {node} failed")

    def kernel_specific_preprocess_data(self, df: pd.DataFrame):
        if S_PARAMETERS_GENERATOR_VERSION in df.columns:
            return df.drop(columns=[S_PARAMETERS_GENERATOR_VERSION])
        return df

    def preprocess_modeling_data(self, node='node-15', metric=S_RUNTIME):
        return self._preprocess_data(mode=S_MODELING, node=node, metric=metric)

    def preprocess_characterization_data(self, node='node-15', metric=S_RUNTIME):
        return self._preprocess_data(mode=S_CHARACTERIZATION, node=node, metric=metric)

    def _preprocess_data(self, node, mode=S_MODELING, metric=S_RUNTIME):
        attr = DataAttributes(node=node, kernel=self.kernel, category=mode, metric=metric, return_as='df')
        df = DataHandler.get_kernel_data(attr)
        df = self.kernel_specific_preprocess_data(df)

        # Scale and transform label values
        # --------------------------------------------------------------------------------------------------------------
        for label in ORIGINAL_LABELS:
            if label in df.columns:
                Transformers.log_transform(df, label)

        df.drop_duplicates(keep='first', inplace=True)

        return df

    @staticmethod
    def select_labels(df: pd.DataFrame, keep_labels=None):
        """ This method filters out all labels and keeps only the ones we are interested in from training and testing
        data """
        if keep_labels is None:
            keep_labels = [S_RUNTIME_MS]

        if type(keep_labels) is not list:
            keep_labels = [keep_labels]

        discarded_labels = []
        for label in ALL_LABELS:
            if label not in keep_labels and label in df.columns:
                discarded_labels.append(label)

        df.drop(columns=discarded_labels, inplace=True)

    def read_characterization_data_summary(self, nodes, overwrite_summary=False, keep_labels=None):

        self.parse_and_combine_characterization_data(nodes, override=overwrite_summary)
        df_dict = dict()
        for node in nodes:
            df: pd.DataFrame = self.preprocess_characterization_data(node=node)
            BaseModel.select_labels(df, keep_labels)
            df_dict[node] = df
        return df_dict

    def read_modeling_data_summary(self, nodes, overwrite_summary=False, keep_labels=None):

        self.parse_and_combine_modeling_data(nodes, override=overwrite_summary)
        df_dict = dict()
        for node in nodes:
            df = self.preprocess_modeling_data(node=node)
            BaseModel.select_labels(df, keep_labels)
            df_dict[node] = df
        return df_dict

    def optimize_model_hyper_parameters(self, training_nodes=None, testing_nodes=None,
                                        override_modeling_data_summary=False, target_label=S_RUNTIME_MS,
                                        stats_only=False):

        nodes = training_nodes + testing_nodes
        metric = get_metric(target_label)

        self.parse_and_combine_modeling_data(nodes=nodes, override=override_modeling_data_summary, metric=metric)
        df_train = pd.DataFrame()
        df_test = pd.DataFrame()
        for node in nodes:
            tmp_df = self.preprocess_modeling_data(node=node, metric=metric)
            node_id = int(node.split('-')[1])
            tmp_df[S_NODE_ID] = [node_id for _ in range(tmp_df.shape[0])]
            if node in testing_nodes:
                logger.info(f"[HYPER_PARAMETER_TUNING] Preprocess testing {node}")
                df_test = df_test.append(tmp_df)
            else:
                logger.info(f"[HYPER_PARAMETER_TUNING] Preprocess training {node}")
                df_train = df_train.append(tmp_df)

        training_groups = df_train[S_NODE_ID]
        df_train.drop(columns=[S_NODE_ID], inplace=True)
        df_test.drop(columns=[S_NODE_ID], inplace=True)
        logger.info(f"[HYPER_PARAMETER_TUNING] Keeping {target_label} as labels")
        BaseModel.select_labels(df_train, keep_labels=target_label)
        BaseModel.select_labels(df_test, keep_labels=target_label)

        logger.info(f"f[HYPER_PARAMETER_TUNING] Training data shape:  {df_train.shape}")
        logger.info(f"f[HYPER_PARAMETER_TUNING] Testing data shape:   {df_test.shape}")

        if stats_only:
            return

        cv_inner = LeaveOneGroupOut()

        params_space = cs.ConfigurationSpace(seed=1)

        parameter = csh.UniformIntegerHyperparameter(name='max_depth', lower=5, upper=12, log=False)
        params_space.add_hyperparameter(parameter)

        parameter = csh.UniformIntegerHyperparameter(name='min_child_weight', lower=5, upper=9, log=False)
        params_space.add_hyperparameter(parameter)

        parameter = csh.UniformFloatHyperparameter(name='subsample', lower=0.8, upper=1.0, log=False)
        params_space.add_hyperparameter(parameter)

        parameter = csh.UniformFloatHyperparameter(name='colsample_bytree', lower=0.8, upper=1.0, log=False)
        params_space.add_hyperparameter(parameter)

        parameter = csh.UniformFloatHyperparameter(name='learning_rate', lower=0.005, upper=0.1, log=True)
        params_space.add_hyperparameter(parameter)

        parameter = csh.UniformFloatHyperparameter(name='reg_alpha', lower=1e-4, upper=1e-2, log=True)
        params_space.add_hyperparameter(parameter)

        parameter = csh.UniformFloatHyperparameter(name='reg_lambda', lower=1e-4, upper=1e-2, log=True)
        params_space.add_hyperparameter(parameter)

        parameter = csh.Constant('max_bin', value=512)
        params_space.add_hyperparameter(parameter)

        parameter = csh.Constant('gamma', value=0)
        params_space.add_hyperparameter(parameter)

        x_train = df_train.drop(columns=[target_label])
        y_train = df_train[target_label]

        model = xgb.XGBRegressor(tree_method='hist', verbosity=0, random_state=0, booster='gbtree')

        # TODO: fix that and the iterations in the method call.
        max_budget = DEFAULT_NUM_BOOST_ROUNDS
        min_budget = MIN_BUDGET
        iterations = DEFAULT_TUNING_ITERATIONS,
        search = HpBandSterSearchCV(model, params_space, resource_name='n_estimators', random_state=0,
                                    n_jobs=N_CPU_CORES, cv=cv_inner, optimizer='bohb',
                                    min_budget=min_budget, n_iter=iterations,
                                    max_budget=max_budget, resource_type=int, verbose=0,
                                    scoring={'custom': Metrics.percentage_less_than_5},
                                    refit='custom').fit(x_train, y_train, groups=training_groups)

        logger.info(f"[HYPER_PARAMETER_TUNING] Best model parameters:  {str(search.best_params_)}")

        # Evaluate the best model on testing data
        x_test = df_test.drop(columns=[target_label])

        y_test = df_test[target_label]

        percentage_error = Metrics.percentage_less_than_5(search, x_test, y_test)
        logger.info(f"[HYPER_PARAMETER_TUNING] Percentage of data with errors less than 5% is: {percentage_error}\n")
        return search.best_params_

    def train_and_evaluate_model_with_tuned_parameters(self, training_nodes=None, testing_nodes=None,
                                                       best_num_boost_round=None, best_params=None,
                                                       target_label=S_RUNTIME_MS, save_model=True,
                                                       overwrite_data_summary=False, timer: StopWatch = None):
        """ This method can be used directly for one of two purposes:
        1- Train and evaluate a model using best parameters found from a previous hyper-parameter tuning stage, or
        2- Find the best number of estimators using early stopping rounds and CV, then apply step 1.

        For scenario 1, best_num_boost_round must be set to a value, else scenario 2 is run.
        """
        nodes = training_nodes + testing_nodes
        metric = get_metric(target_label)
        self.parse_and_combine_modeling_data(nodes=nodes, override=overwrite_data_summary, metric=metric)

        if best_params is None:
            logger.info("[TRAINING_AND_EVALUATION] No model parameters supplied, using default")
            best_params = {'colsample_bytree': 0.8, 'learning_rate': 0.05, 'max_depth': 10, 'min_child_weight': 7,
                           'n_estimators': 3000, 'reg_alpha': 0.001, 'reg_lambda': 0.001, 'subsample': 1.0}
        else:
            logger.info("Using best parameters from previous hyper-tuning phase")
            logger.info(f"[TRAINING_AND_EVALUATION] Best params: {best_params}")

        df_train = pd.DataFrame()
        df_test = pd.DataFrame()

        for node in nodes:
            logger.info(f"[TRAINING_AND_EVALUATION] Preprocess data of training {node}")
            tmp_df = self.preprocess_modeling_data(node=node, metric=metric)
            node_id = int(node.split('-')[1])
            tmp_df[S_NODE_ID] = [node_id for _ in range(tmp_df.shape[0])]
            if node in testing_nodes:
                df_test = df_test.append(tmp_df)
            else:
                df_train = df_train.append(tmp_df)
        training_groups = df_train[S_NODE_ID]

        df_train.drop(columns=[S_NODE_ID], inplace=True)
        df_test.drop(columns=[S_NODE_ID], inplace=True)

        logger.info(f"[TRAINING_AND_EVALUATION] Keeping {target_label} as labels")

        BaseModel.select_labels(df_train, keep_labels=target_label)
        BaseModel.select_labels(df_test, keep_labels=target_label)

        params = dict()
        params['objective'] = 'reg:squarederror'
        params['eval_metric'] = ['rmse']
        params['tree_method'] = 'hist'

        num_boost_round = best_params.get('n_estimators', None)
        if num_boost_round is not None:
            best_params.pop('n_estimators')

        params.update(best_params)
        params['nthread'] = N_CPU_CORES

        x_train = df_train.drop(columns=[target_label])
        y_train = df_train[target_label]
        d_train = xgb.DMatrix(x_train, label=y_train)

        x_test = df_test.drop(columns=[target_label])
        y_test = df_test[target_label]
        d_test = xgb.DMatrix(x_test, label=y_test)

        logger.info(f"[TRAINING_AND_EVALUATION] Train data shape {x_train.shape}")
        logger.info(f"[TRAINING_AND_EVALUATION] Test data shape {x_test.shape}")

        best_cv_results = None
        early_stopping_rounds = EARLY_STOPPING_ROUNDS
        if best_num_boost_round is None:
            cv_inner = LeaveOneGroupOut()
            if num_boost_round is None:
                logger.info(
                    f"[TRAINING_AND_EVALUATION] Hyper-tuning phase did not yield n_estimators, using default of "
                    f"{DEFAULT_NUM_BOOST_ROUNDS}")
                num_boost_round = DEFAULT_NUM_BOOST_ROUNDS
            # Train with CV to find best number of boosting rounds (number of estimators)
            best_cv_results, best_num_boost_round = \
                xgb_cv(params, num_boost_round, cv_inner, training_groups, d_train, x_train, None,
                       early_stopping_rounds=early_stopping_rounds)

        # Train final model
        model = xgb.train(params, d_train, num_boost_round=best_num_boost_round)

        # Evaluate
        y_pred = model.predict(d_test)
        testing_eval_results = Metrics.XGBoostEval.percentage_error_from_dmatrix(y_pred, d_test)
        # Convert values to strings because serializing to json returns error for non primitive types
        testing_eval_results_dict = dict()
        for metric_tuple in testing_eval_results:
            testing_eval_results_dict[metric_tuple[0]] = str(metric_tuple[1])

        # Report training results
        y_pred = model.predict(d_train)
        training_eval_results = Metrics.XGBoostEval.percentage_error_from_dmatrix(y_pred, d_train)
        training_eval_results_dict = dict()
        for metric_tuple in training_eval_results:
            training_eval_results_dict[metric_tuple[0]] = str(metric_tuple[1])

        # Create meta data for this model
        meta_data = dict()
        meta_data['kernel'] = self.kernel
        meta_data['label'] = target_label
        meta_data['num_boost_rounds'] = str(best_num_boost_round)
        meta_data['model_params'] = params
        meta_data['timestamp'] = run_timestamp
        meta_data['training_nodes'] = [i for i in nodes if i not in testing_nodes]
        meta_data['testing_nodes'] = testing_nodes
        meta_data['evaluation_results'] = testing_eval_results_dict
        meta_data['training_results'] = training_eval_results_dict
        meta_data['evaluation_function'] = Metrics.XGBoostEval.percentage_error_from_dmatrix.__name__
        meta_data['training_data_shape'] = x_train.shape
        meta_data['testing_data_shape'] = x_test.shape
        meta_data['log_transformation'] = LOG_TRANSFORM
        meta_data['early_stopping_limit'] = early_stopping_rounds
        meta_data['training_time_seconds'] = timer.elapsed_s()
        if best_cv_results is not None:
            meta_data['best_cv_results'] = best_cv_results.to_dict()

        # Save the trained model and its meta-data
        model_file_name = f'xgb_{target_label}_{run_timestamp}.model'
        model_metadata_file_name = f'xgb_meta_{target_label}_{run_timestamp}.json'
        model_dir_path = join_path(SAVED_MODELS_DIR, self.kernel, HOSTNAME, run_timestamp)
        model_file_path = join_path(model_dir_path, model_file_name)
        model_metadata_file_path = join_path(model_dir_path, model_metadata_file_name)

        os.makedirs(model_dir_path, exist_ok=True)
        if save_model:
            model.save_model(model_file_path)
            FileUtils.serialize(meta_data, file_path=model_metadata_file_path, file_extension='json')

    def re_evaluate_for_reporting(self, hour_timestamp, load_model=True):
        """ This method loads an existing kernel model or retrains a kernel model using parameters from a JSON
        output file (of a previous training trial). """
        # Search for the folder path
        saved_models_dir = join_path(SAVED_MODELS_DIR, self.kernel)
        saved_model_dir = None
        for node_data_path in os.listdir(saved_models_dir):
            if hour_timestamp in os.listdir(join_path(saved_models_dir, node_data_path)):
                saved_model_dir = join_path(saved_models_dir, node_data_path, hour_timestamp)
        if saved_model_dir is None:
            logger.info("Couldn't find the model directory path. Exiting.")

        # Search for the meta-data file
        meta_data_file_path = None
        for file_name in os.listdir(saved_model_dir):
            if file_name.__contains__('meta'):
                meta_data_file_path = join_path(saved_model_dir, file_name)
                break
        if meta_data_file_path is None:
            return

        with open(meta_data_file_path) as file_obj:
            obj = js.load(file_obj)
            # Set the global target label accordingly
            # global target_label
            target_label = obj['label']
            testing_nodes = obj['testing_nodes']
            training_nodes = obj['training_nodes']
            log_transform = bool(obj['log_transformation'])
            model_params = obj['model_params']
            num_boost_rounds = int(obj['num_boost_rounds'])

            logger.info(f"Label:              {target_label}")
            logger.info(f"Kernel:             {self.kernel}")
            logger.info(f"# Boost rounds:     {num_boost_rounds}")
            logger.info(f"Training nodes:     {training_nodes}")
            logger.info(f"Testing nodes:      {testing_nodes}")
            logger.info(f"Log transform:      {log_transform}")

        df_test = pd.DataFrame()
        for node in testing_nodes:
            logger.info(f"[RE-EVALUATION] Preprocess data of {node}")
            tmp_df = self.preprocess_modeling_data(node=node)
            df_test = df_test.append(tmp_df)
        BaseModel.select_labels(df_test, keep_labels=target_label)
        x_test = df_test.drop(columns=[target_label])
        y_test = df_test[target_label]
        d_test = xgb.DMatrix(x_test, label=y_test)

        df_train = pd.DataFrame()
        for node in training_nodes:
            logger.info(f"[RE-EVALUATION] Preprocess data of {node}")
            tmp_df = self.preprocess_modeling_data(node=node)
            df_train = df_train.append(tmp_df)
        BaseModel.select_labels(df_train, keep_labels=target_label)

        x_train = df_train.drop(columns=[target_label])
        y_train = df_train[target_label]
        d_train = xgb.DMatrix(x_train, label=y_train)

        if load_model:
            model_file_path = None
            for file_name in os.listdir(saved_model_dir):
                if file_name.__contains__('model'):
                    model_file_path = join_path(saved_model_dir, file_name)
                    break

            if model_file_path is None:
                return

            # Re-evaluate the model to report additional error stats
            model = xgb.Booster()
            model.load_model(model_file_path)

        else:

            model = xgb.train(
                model_params,
                d_train,
                num_boost_round=num_boost_rounds,
                verbose_eval=True
            )

        logger.info()
        # Predict and evaluate test nodes
        y_pred = model.predict(d_test)

        logger.info("Testing data")
        results = Metrics.XGBoostEval.percentage_error_from_dmatrix(y_pred, d_test)
        for key, val in results:
            if key in ['real-rmse', 'real-mae']:
                logger.info(f"{key.split('-')[1]}: {val}")
            else:
                logger.info(f"{key}: {val}")
        logger.info("________________________________________________________________")
        return meta_data_file_path

    def create_model(self, train_nodes, test_nodes, model_labels, overwrite_data_summary, stats_only=False):
        # Timer to measure how long it takes to train models
        training_timer = StopWatch()
        training_timer.start()
        best_params = self.optimize_model_hyper_parameters(training_nodes=train_nodes, testing_nodes=test_nodes,
                                                           target_label=model_labels,
                                                           override_modeling_data_summary=overwrite_data_summary,
                                                           stats_only=stats_only)
        if not stats_only:
            # We set overwrite here to false because we do it already in the optimization phase
            self.train_and_evaluate_model_with_tuned_parameters(training_nodes=train_nodes, testing_nodes=test_nodes,
                                                                best_params=best_params, target_label=model_labels,
                                                                overwrite_data_summary=False,
                                                                timer=training_timer)

    def preprocess_kernel_parameters(self, kernel_parameters: BaseParameters, return_type=dict):
        df = self.kernel_specific_preprocess_data(kernel_parameters.to_df())
        if return_type == dict:
            return df.to_dict(orient='records')[0]
        else:
            return df


class ConvModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.kernel = 'conv2d'

    def kernel_specific_preprocess_data(self, df):
        df = super().kernel_specific_preprocess_data(df)
        # df["group_ratio"] = df[Conv2dNames.in_channels] / df[Conv2dNames.groups]
        # Drop unneeded columns

        # --------------------------------------------------------------------------------------------------------------
        df.drop(columns=[Conv2dNames.n], inplace=True)
        # df.drop(columns=[Conv2dNames.groups], inplace=True)
        df.drop(columns=[Conv2dNames.c], inplace=True)
        df.drop(columns=[Conv2dNames.padding_mode], inplace=True)
        # h = w for all these parameters so we drop redundant info

        # df['kernel_size'] = df[Conv2dNames.kernel_h] * df[Conv2dNames.kernel_w]
        # df['input_size'] = df[Conv2dNames.in_channels] * (df[Conv2dNames.h] + df[Conv2dNames.padding_h]) * \
        #                    (df[Conv2dNames.w] + df[Conv2dNames.padding_w])
        # df['stride_size'] = df[Conv2dNames.stride_h] * df[Conv2dNames.stride_w]

        # df.drop(columns=[Conv2dNames.kernel_h, Conv2dNames.h, Conv2dNames.stride_h, Conv2dNames.padding_h],
        #         inplace=True)

        df.drop(columns=[Conv2dNames.w, Conv2dNames.stride_w, Conv2dNames.padding_w, Conv2dNames.kernel_w,
                         Conv2dNames.dilation_w], inplace=True)

        # df.drop(columns=[Conv2dNames.bias, Conv2dNames.stride_h, Conv2dNames.stride_w,
        #                  Conv2dNames.padding_h, Conv2dNames.padding_w, Conv2dNames.dilation_h,
        #                  Conv2dNames.dilation_w], inplace=True)

        return df


class BatchNormModel(BaseModel):

    def __init__(self):
        super().__init__()
        self.kernel = 'batchnorm2d'

    def kernel_specific_preprocess_data(self, df: pd.DataFrame):
        df = super().kernel_specific_preprocess_data(df)
        # C is the same as number of features which is left as is, so we remove C
        df.drop(columns=[BatchNormNames.n, BatchNormNames.c], inplace=True)
        return df


class MatMulModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.kernel = 'matmul'

    def kernel_specific_preprocess_data(self, df):
        df = super().kernel_specific_preprocess_data(df)
        df.drop(columns=[MatMulAddNames.mat2_r, MatMulAddNames.bias, MatMulAddNames.bias_size, MatMulAddNames.mat1_r],
                inplace=True)
        return df


class MaxPoolModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.kernel = 'maxpool2d'

    def kernel_specific_preprocess_data(self, df):
        df = super().kernel_specific_preprocess_data(df)
        df.drop(columns=[MaxPool2dNames.return_indices, MaxPool2dNames.dilation, MaxPool2dNames.n], inplace=True)
        return df


class AdaptivePoolModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.kernel = 'adaptivepool2d'

    def kernel_specific_preprocess_data(self, df):
        df = super().kernel_specific_preprocess_data(df)
        df.drop(columns=[AdaptivePool2dNames.n], inplace=True)
        return df


class ReluModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.kernel = 'relu'

    def kernel_specific_preprocess_data(self, df):
        df = super().kernel_specific_preprocess_data(df)
        df.drop(columns=[ReluNames.n], inplace=True)
        return df


class CatModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.kernel = 'cat'

    def kernel_specific_preprocess_data(self, df):
        return super().kernel_specific_preprocess_data(df)


class AddModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.kernel = 'add'

    def kernel_specific_preprocess_data(self, df):
        df = super().kernel_specific_preprocess_data(df)
        df.drop(columns=[AddNames.n, AddNames.w], inplace=True)
        return df
