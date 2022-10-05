import os
import traceback
from os.path import join as join_path

import ConfigSpace as cs
import ConfigSpace.hyperparameters as csh
import pandas as pd
import xgboost as xgb
from hpbandster_sklearn import HpBandSterSearchCV
from sklearn.model_selection import LeaveOneGroupOut, KFold

from pyutils.analysis.common.data_transformers import Transformers
from pyutils.analysis.common.error_metrics import Metrics, MetricPrettyName
from pyutils.common.config import ALL_LABELS, ORIGINAL_LABELS, N_CPU_CORES, MINUTE_TIMESTAMP
from pyutils.common.data_handlers.data_interface import DataAttributes, DataHandler
from pyutils.common.methods import get_metric
from pyutils.common.paths import SAVED_MODELS_DIR
from pyutils.common.strings import S_NODE_ID, S_NETWORK, S_RUNTIME_MS, S_RUNTIME, S_CHARACTERIZATION
from pyutils.common.timers import StopWatch
from pyutils.common.utils import FileUtils
from pyutils.common.utils import GlobalLogger
from pyutils.hosts.common import HOSTNAME
from pyutils.modeling.config import DEFAULT_NUM_BOOST_ROUNDS, EARLY_STOPPING_ROUNDS, LOG_TRANSFORM, \
    DEFAULT_TUNING_ITERATIONS, MIN_BUDGET

logger = GlobalLogger().get_logger()

run_timestamp = MINUTE_TIMESTAMP
lowest_permitted_cpu_freq = 422400


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
    # logger.info(f"[TRAINING_AND_EVALUATION] Validation set best results:
    # {str(cv_results.iloc[best_boost_rounds_idx])}")
    # logger.info(f"[TRAINING_AND_EVALUATION] End of validation results.")
    return cv_results.iloc[best_boost_rounds_idx], best_num_boost_rounds


class NetworksBaseModel:
    def __init__(self):
        self.network = "BASE_MODEL_HAS_NO_NETWORK"
        self.evaluation_function = Metrics.percentage_less_than_5

    def parse_and_combine_characterization_data(self, nodes=None, overwrite=False, metric=S_RUNTIME):
        return self._parse_and_combine_experiments_data(mode=S_CHARACTERIZATION, nodes=nodes, override=overwrite,
                                                        metric=metric)

    def _parse_and_combine_experiments_data(self, mode=S_CHARACTERIZATION, nodes=None, override=False,
                                            metric=S_RUNTIME):
        """ The method parses the data and saves the files, so no need to return anything. """

        logger.info(f"Parsing {mode} data for: {self.network}")
        for node in nodes:
            logger.info(f"-- Processing {node} - Metric: {metric}")
            attr = DataAttributes(node=node, benchmark=self.network, category=mode, metric=metric,
                                  overwrite_summary=override)
            try:
                DataHandler.get_benchmark_data(attr)
            except Exception as e:
                st_trace = traceback.format_exc()
                logger.error(e)
                logger.error(st_trace)
                logger.error(f"Parsing data from {node} failed")

    def preprocess_characterization_data(self, node='node-15', metric=S_RUNTIME):
        return self._preprocess_data(node=node, metric=metric)

    def _preprocess_data(self, node, metric=S_RUNTIME):
        attr = DataAttributes(node=node, benchmark=self.network, category=S_CHARACTERIZATION, metric=metric,
                              return_as='df')
        df = DataHandler.get_benchmark_data(attr)

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
            raise Exception(f"Specify labels: {keep_labels}")

        if type(keep_labels) is not list:
            keep_labels = [keep_labels]

        discarded_labels = []
        for label in ALL_LABELS:
            if label not in keep_labels and label in df.columns:
                discarded_labels.append(label)

        df.drop(columns=discarded_labels, inplace=True)

    def read_characterization_data_summary(self, nodes, overwrite_summary=False, keep_labels=None):

        self.parse_and_combine_characterization_data(nodes, overwrite=overwrite_summary)
        df_dict = dict()
        for node in nodes:
            df: pd.DataFrame = self.preprocess_characterization_data(node=node)
            NetworksBaseModel.select_labels(df, keep_labels)
            df_dict[node] = df
        return df_dict

    def optimize_model_hyper_parameters(self, training_nodes=None, testing_nodes=None,
                                        overwrite_data_summary=False, target_label=S_RUNTIME_MS, stats_only=False):

        nodes = training_nodes + testing_nodes
        metric = get_metric(target_label)

        self.parse_and_combine_characterization_data(nodes=nodes, overwrite=overwrite_data_summary, metric=metric)
        df_train = pd.DataFrame()
        df_test = pd.DataFrame()
        for node in nodes:
            tmp_df = self.preprocess_characterization_data(node=node, metric=metric)
            node_id = int(node.split('-')[1])
            tmp_df[S_NODE_ID] = [node_id for _ in range(tmp_df.shape[0])]
            if node in testing_nodes:
                logger.info(f"[HYPER_PARAMETER_TUNING] Preprocess testing {node} - {tmp_df.shape[0]}")
                df_test = df_test.append(tmp_df)
            else:
                logger.info(f"[HYPER_PARAMETER_TUNING] Preprocess training {node} - {tmp_df.shape[0]}")
                df_train = df_train.append(tmp_df)

        training_groups = df_train[S_NODE_ID]
        df_train.drop(columns=[S_NODE_ID], inplace=True)
        df_test.drop(columns=[S_NODE_ID], inplace=True)

        logger.info(f"[HYPER_PARAMETER_TUNING] Keeping {target_label} as labels")
        NetworksBaseModel.select_labels(df_train, keep_labels=target_label)
        NetworksBaseModel.select_labels(df_test, keep_labels=target_label)

        logger.info(f"f[HYPER_PARAMETER_TUNING] Training data shape:  {df_train.shape}")
        logger.info(f"f[HYPER_PARAMETER_TUNING] Testing data shape:   {df_test.shape}")

        if stats_only:
            return

        cv_inner = LeaveOneGroupOut()
        if len(training_nodes) == 1:
            cv_inner = KFold(5, shuffle=True, random_state=0)
            training_groups = None

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

        max_budget = DEFAULT_NUM_BOOST_ROUNDS
        min_budget = MIN_BUDGET

        # HpBandSter always finds the set of parameters that maximizes the score
        search = HpBandSterSearchCV(model, params_space, resource_name='n_estimators', random_state=0,
                                    n_jobs=N_CPU_CORES, cv=cv_inner, optimizer='bohb',
                                    min_budget=min_budget,
                                    max_budget=max_budget, resource_type=int,
                                    n_iter=DEFAULT_TUNING_ITERATIONS, verbose=0,
                                    scoring={'custom': self.evaluation_function},  # Always maximized
                                    refit='custom').fit(x_train, y_train, groups=training_groups)

        logger.info(f"[HYPER_PARAMETER_TUNING] Best model parameters:  {str(search.best_params_)}")
        best_params = search.best_params_

        # Evaluate the best model on testing data
        x_test = df_test.drop(columns=[target_label])

        y_test = df_test[target_label]

        percentage_error = Metrics.percentage_less_than_5(search, x_test, y_test)
        logger.info(f"[HYPER_PARAMETER_TUNING] Percentage of data with errors less than 5% is: {percentage_error}\n")
        return best_params

    def train_and_evaluate_model_with_tuned_parameters(self, training_nodes=None, testing_nodes=None,
                                                       best_num_boost_round=None, best_params=None,
                                                       target_label=S_RUNTIME_MS, save_model=True,
                                                       overwrite_data_summary=False, timer: StopWatch = None,
                                                       prefix=None):
        """ This method can be used directly for one of two purposes:
        1- Train and evaluate a model using best parameters found from a previous hyper-parameter tuning stage, or
        2- Find the best number of estimators using early stopping rounds and CV, then apply step 1.

        For scenario 1, best_num_boost_round must be set to a value, else scenario 2 is run.
        """
        nodes = training_nodes + testing_nodes
        metric = get_metric(target_label)
        self.parse_and_combine_characterization_data(nodes=nodes, overwrite=overwrite_data_summary, metric=metric)

        logger.info("Using best parameters from previous hyper-tuning phase")
        logger.info(f"[TRAINING_AND_EVALUATION] Best params: {best_params}")

        df_train = pd.DataFrame()
        df_test = pd.DataFrame()

        for node in nodes:
            logger.info(f"[TRAINING_AND_EVALUATION] Preprocess data of training {node}")
            tmp_df = self.preprocess_characterization_data(node=node, metric=metric)
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

        NetworksBaseModel.select_labels(df_train, keep_labels=target_label)
        NetworksBaseModel.select_labels(df_test, keep_labels=target_label)

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
            if len(training_nodes) == 1:
                cv_inner = KFold(5, shuffle=True, random_state=0)
                training_groups = None

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
        meta_data[S_NETWORK] = self.network
        meta_data['label'] = target_label
        meta_data['num_boost_rounds'] = str(best_num_boost_round)
        meta_data['model_params'] = params
        meta_data['timestamp'] = run_timestamp
        meta_data['training_nodes'] = [i for i in nodes if i not in testing_nodes]
        meta_data['testing_nodes'] = testing_nodes
        meta_data['evaluation_results'] = testing_eval_results_dict
        meta_data['training_results'] = training_eval_results_dict
        meta_data['evaluation_function'] = self.evaluation_function.__name__
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
        if prefix is None:
            model_dir_path = join_path(SAVED_MODELS_DIR, self.network, HOSTNAME, run_timestamp)
        else:
            model_dir_path = join_path(SAVED_MODELS_DIR, prefix, self.network, HOSTNAME, run_timestamp)
        os.makedirs(model_dir_path, exist_ok=True)

        model_file_path = join_path(model_dir_path, model_file_name)
        model_metadata_file_path = join_path(model_dir_path, model_metadata_file_name)

        if save_model:
            model.save_model(model_file_path)
            FileUtils.serialize(meta_data, file_path=model_metadata_file_path, file_extension='json')

        logger.info("Evaluation results: ")
        logger.info(MetricPrettyName.get_pretty_evaluation_metrics(testing_eval_results_dict))

    def create_model(self, train_nodes, test_nodes, model_labels, overwrite_data_summary, prefix=None,
                     stats_only=False):
        # Timer to measure how long it takes to train models
        training_timer = StopWatch()
        training_timer.start()
        best_params = self.optimize_model_hyper_parameters(training_nodes=train_nodes, testing_nodes=test_nodes,
                                                           target_label=model_labels,
                                                           overwrite_data_summary=overwrite_data_summary,
                                                           stats_only=stats_only)

        if not stats_only:
            # We set overwrite here to false because we do it already in the optimization phase
            self.train_and_evaluate_model_with_tuned_parameters(training_nodes=train_nodes, testing_nodes=test_nodes,
                                                                best_params=best_params, target_label=model_labels,
                                                                overwrite_data_summary=False,
                                                                timer=training_timer, prefix=prefix)
