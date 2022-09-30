# TODO: unused code for the time being.

# 1- train on node-14 vs 1 to 12, then test on 2,13
# 2- collect the maximum number of common records between node-14 and nodes 1 to 12 (or any others that maximize this
# intersection.
# 3- in a CV fashion, collect max number of common records between node-14 and 11 nodes and test on the remaining 2,
# where these 11 and 2 nodes are changed, then we report the summary of performance to show that it is not only luck.


# Design for 3, test 2, and evaluate 1 as an extra reference point.
# A common config is a config that exists in node-14 data, and any other of the training nodes
import argparse
import os
import random
from itertools import combinations
from os.path import join as jp

import ConfigSpace as cs
import ConfigSpace.hyperparameters as csh
import pandas as pd
import xgboost as xgb
from hpbandster_sklearn import HpBandSterSearchCV
from pandas import DataFrame
from pyutils.modeling.kernels_training_models import ConvModel, xgb_cv
from sklearn.model_selection import LeaveOneGroupOut

from pyutils.analysis.common.data_transformers import Transformers
from pyutils.analysis.common.error_metrics import Metrics
from pyutils.analysis.one_vs_many.config import MANY_DATA_DIR, ONE_DATA_DIR, COMMON_DATA_DIR
from pyutils.common.config import N_CPU_CORES, \
    MINUTE_TIMESTAMP
from pyutils.common.data_interface import DataAttributes, DataHandler
from pyutils.common.strings import S_NODE_ID, S_RUNTIME_MS, S_RUNTIME, S_MODELING
from pyutils.common.utils import FileUtils as fu, GlobalLogger
from pyutils.hosts.common import HOSTNAME
from pyutils.modeling.config import DEFAULT_NUM_BOOST_ROUNDS, LOG_TRANSFORM, MIN_BUDGET, DEFAULT_TUNING_ITERATIONS

logger = GlobalLogger().get_logger()

X_NODE_ID = f'{S_NODE_ID}_x'
Y_NODE_ID = f'{S_NODE_ID}_y'

X_RUNTIME_MS = f'{S_RUNTIME_MS}_x'
Y_RUNTIME_MS = f'{S_RUNTIME_MS}_y'
run_timestamp = MINUTE_TIMESTAMP

REFERENCE_NODE = 'node-02'


class Utilities:
    @staticmethod
    def create_triplets(nodes):
        # Note: the input list must have unique elements
        _testing_nodes = list()
        for comb in combinations(nodes, 3):
            _testing_nodes.append([i for i in comb])
        return _testing_nodes

    @staticmethod
    def get_single_node_data(node=REFERENCE_NODE, overwrite=False):
        data_file_path = jp(ONE_DATA_DIR, f'{node}_data.csv')
        if os.path.exists(data_file_path) and not overwrite:
            return pd.read_csv(data_file_path)

        if os.path.exists(data_file_path):
            os.remove(data_file_path)

        one_attr = DataAttributes(node=node, kernel='conv2d', category=S_MODELING, metric=S_RUNTIME,
                                  overwrite_summary=False,
                                  aggregate=False, return_as='df')
        one_data: DataFrame = DataHandler.get_kernel_data(one_attr)
        one_data[S_NODE_ID] = node
        one_data.to_csv(data_file_path, index=False)

        conv_model = ConvModel()
        one_data = conv_model.kernel_specific_preprocess_data(one_data)

        logger.info(f"Single node data size: {one_data.shape[0]}")
        return one_data

    @staticmethod
    def get_multiple_nodes_data(nodes):
        # TODO: for sanity, fix later
        overwrite = True
        data_file_path = jp(MANY_DATA_DIR, f'many_nodes_data.csv')
        fu.silent_remove(data_file_path)
        tmp = list()
        for node in nodes:
            attr = DataAttributes(node=node, kernel='conv2d', category=S_MODELING, metric=S_RUNTIME,
                                  overwrite_summary=False,
                                  aggregate=False, return_as='df')
            node_data: DataFrame = DataHandler.get_kernel_data(attr)
            node_data[S_NODE_ID] = node
            tmp.append(node_data)
            logger.debug(f"{node} data size is: {len(node_data)}")

        combined_nodes_data = pd.concat(tmp, ignore_index=True)
        # We reduce the number of features for faster comparisons by removing the columns that we do not need in the
        # modeling phase anyway
        conv_model = ConvModel()
        combined_nodes_data = conv_model.kernel_specific_preprocess_data(combined_nodes_data)

        logger.info(f"Multiple nodes are: {nodes}")
        logger.info(f" -- data size is: {combined_nodes_data.shape}")
        combined_nodes_data.to_csv(data_file_path, index=False)
        return combined_nodes_data

    @staticmethod
    def get_common_configurations(single_node: str, training_nodes: list, overwrite=False):
        data_file_path = jp(COMMON_DATA_DIR, f'{single_node}_common.csv')
        if os.path.exists(data_file_path) and not overwrite:
            common = pd.read_csv(data_file_path)
        else:
            fu.silent_remove(data_file_path)

            single_node_data = Utilities.get_single_node_data(node=single_node, overwrite=overwrite)
            multiple_nodes_data = Utilities.get_multiple_nodes_data(nodes=training_nodes)

            # Merge on the features columns
            cols = single_node_data.columns
            common = pd.merge(single_node_data, multiple_nodes_data, on=[i for i in cols if i not in
                                                                         [S_RUNTIME_MS, S_NODE_ID]])

            # Append single node data to construct the combined training data
            single_node_data.rename(columns={S_RUNTIME_MS: X_RUNTIME_MS, S_NODE_ID: X_NODE_ID}, inplace=True)
            single_node_data[Y_RUNTIME_MS] = single_node_data[X_RUNTIME_MS]
            single_node_data[Y_NODE_ID] = single_node_data[X_NODE_ID]

            common = common.append(single_node_data)
            common.sort_values(by=[f'{S_NODE_ID}_y'], inplace=True)
            common.to_csv(data_file_path, index=False)
        logger.info(f"Common data size: {common.shape}")
        return common

    @staticmethod
    def get_number_of_intersections_with_node(node, nodes):
        data = Utilities.get_common_configurations(training_nodes=nodes, single_node=node)
        for node in nodes:
            count = data[data[f'{S_NODE_ID}_y'] == node].shape[0]
            print(node, count)


class Modeling:
    @staticmethod
    def median_of_common(df):
        # df[f'{S_NODE_ID}_y'] = 'many'

        grouped = df.groupby(by=[i for i in df.columns if i not in [Y_NODE_ID, X_RUNTIME_MS, Y_RUNTIME_MS, X_NODE_ID]])
        new_df = pd.DataFrame()
        for name, group in grouped:
            # x = group[[Y_RUNTIME_MS]].median()
            x = group[[Y_RUNTIME_MS]].mean()
            group[Y_RUNTIME_MS] = float(x)
            # print(name)
            # print('--------------------')
            # group.drop_duplicates(inplace=True, subset=[i for i in df.columns if
            #                                             i not in [Y_NODE_ID, X_RUNTIME_MS, Y_RUNTIME_MS, X_NODE_ID]])
            # print(group)
            # print(group.to_dict())
            new_df = new_df.append(group)
            # print('--------------------')
            # print(x)
            #
            # print(name, group, x)
            # break
        return new_df

    @staticmethod
    def train_node_model(training_nodes: list, test_nodes: list, single_node: str, mode='single', overwrite=False,
                         port=9090, n_jobs=-1):
        # We remove the single node from training nodes because merge behavior when it finds three similar records
        # is to pick only two. So we get the common records between the single node and unique training nodes, then
        # we append the single node to the result to end up with the global training data containing single node records
        # plus whatever intersection it found in the other records.

        target_label = Y_RUNTIME_MS

        # Prepare training data
        training_nodes.remove(single_node)
        df_train = Utilities.get_common_configurations(single_node=single_node, training_nodes=training_nodes,
                                                       overwrite=overwrite)
        for label in [Y_RUNTIME_MS, X_RUNTIME_MS]:
            if label in df_train.columns:
                Transformers.log_transform(df_train, label)

        # Prepare testing data
        df_test = Utilities.get_multiple_nodes_data(nodes=test_nodes)
        Transformers.log_transform(df_test, S_RUNTIME_MS)
        df_test.drop(columns=[S_NODE_ID], inplace=True)
        # To use the same target label as the training data
        df_test.rename(columns={S_RUNTIME_MS: Y_RUNTIME_MS}, inplace=True)

        if mode == 'single':
            df_train = df_train[df_train[Y_NODE_ID] == single_node].copy()
            df_train.reset_index(inplace=True, drop=True)

            # Add nodes virtually to re-use the same training groups in the single mode
            num_rows = df_train.shape[0]
            tmp_nodes = [random.choice(training_nodes) for _ in range(num_rows)]
            df_train[Y_NODE_ID] = tmp_nodes

        training_groups = df_train[Y_NODE_ID]

        # Remove the columns we do not need
        if mode == 'single':
            df_train.drop(inplace=True, columns=[Y_NODE_ID, Y_RUNTIME_MS, X_NODE_ID])
        elif mode == 'many':
            df_train.drop(inplace=True, columns=[Y_NODE_ID, X_RUNTIME_MS, X_NODE_ID])
        else:
            raise Exception(f"Unsupported mode {mode}")

        logger.info(f"Training data size: {df_train.shape}")
        logger.info(f"Testing data size:  {df_test.shape}")

        cv_inner = LeaveOneGroupOut()
        groups = training_groups

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

        parameter = csh.Constant('max_bin', value=256)
        params_space.add_hyperparameter(parameter)

        parameter = csh.Constant('gamma', value=0)
        params_space.add_hyperparameter(parameter)

        if mode == 'single':
            x_train = df_train.drop(columns=[X_RUNTIME_MS])
            y_train = df_train[X_RUNTIME_MS]
        else:
            x_train = df_train.drop(columns=[target_label])
            y_train = df_train[target_label]

        model = xgb.XGBRegressor(tree_method='hist', verbosity=0, random_state=0, booster='gbtree')

        max_budget = DEFAULT_NUM_BOOST_ROUNDS
        min_budget = MIN_BUDGET
        iterations = DEFAULT_TUNING_ITERATIONS

        # Exceptional case for node 290 where we run two jobs in parallel usually
        if n_jobs == -1:
            if HOSTNAME.__contains__('node290'):
                n_jobs = int(N_CPU_CORES / 2)
            else:
                n_jobs = N_CPU_CORES

        logger.info(f"Mode is:          {mode}")
        logger.info(f"Number of jobs:   {n_jobs}")
        logger.info(f"Budget limits:    {min_budget}:{max_budget}")
        logger.info(f"Iterations:       {iterations}")
        logger.info(f"Name server port: {port}")

        search = HpBandSterSearchCV(model, params_space, resource_name='n_estimators', random_state=0,
                                    n_jobs=n_jobs, cv=cv_inner, optimizer='bohb',
                                    min_budget=min_budget, max_budget=max_budget, resource_type=int,
                                    n_iter=iterations, verbose=0,
                                    scoring={'custom': Metrics.percentage_less_than_5},
                                    refit='custom', nameserver_host="127.0.0.1",
                                    nameserver_port=int(port)).fit(x_train, y_train, groups=groups)
        best_params = search.best_params_
        logger.info(f"[HYPER_PARAMETER_TUNING] Best model parameters:  {str(best_params)}")

        # Evaluate the best model on testing data
        x_test = df_test.drop(columns=[target_label])
        y_test = df_test[target_label]

        params = dict()
        params['objective'] = 'reg:squarederror'
        params['eval_metric'] = ['rmse']
        params['tree_method'] = 'hist'
        num_boost_round = best_params.get('n_estimators', None)
        if num_boost_round is not None:
            best_params.pop('n_estimators')
        params.update(best_params)
        params['nthread'] = n_jobs

        d_test = xgb.DMatrix(x_test, label=y_test)
        d_train = xgb.DMatrix(x_train, label=y_train)

        if num_boost_round is None:
            logger.info(
                f"[TRAINING_AND_EVALUATION] Hyper-tuning phase did not yield n_estimators, using default of "
                f"{DEFAULT_NUM_BOOST_ROUNDS}")
            num_boost_round = DEFAULT_NUM_BOOST_ROUNDS

        early_stopping_rounds = 50
        logger.info(f"[TRAINING_AND_EVALUATION] Training with parameters: {params}")
        best_cv_results, best_num_boost_round = xgb_cv(params, num_boost_round, cv_inner, training_groups, d_train,
                                                       x_train, groups, early_stopping_rounds=early_stopping_rounds)

        # Train final model
        model = xgb.train(params, d_train, num_boost_round=num_boost_round)
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

        # Log the test nodes evaluation results
        logger.info(f"Testing data evaluation results: {testing_eval_results_dict}")

        # Create meta data for this model
        meta_data = dict()
        meta_data['kernel'] = 'conv2d'
        meta_data['label'] = target_label
        meta_data['num_boost_rounds'] = str(num_boost_round)
        meta_data['model_params'] = params
        meta_data['training_nodes'] = [i for i in training_nodes if i not in test_nodes]
        meta_data['testing_nodes'] = test_nodes
        meta_data['evaluation_results'] = testing_eval_results_dict
        meta_data['training_results'] = training_eval_results_dict
        meta_data['evaluation_function'] = Metrics.XGBoostEval.percentage_error_from_dmatrix.__name__
        meta_data['training_data_shape'] = x_train.shape
        meta_data['testing_data_shape'] = x_test.shape
        meta_data['log_transformation'] = LOG_TRANSFORM
        meta_data['early_stopping_limit'] = early_stopping_rounds
        meta_data['mode'] = mode
        if best_cv_results is not None:
            meta_data['best_cv_results'] = best_cv_results.to_dict()

        # Save the trained model and its meta-data
        model_file_name = f'xgb_{S_RUNTIME_MS}_{run_timestamp}.model'
        model_metadata_file_name = f'xgb_meta_{S_RUNTIME_MS}_{run_timestamp}.json'
        if mode == 'single':
            model_dir_path = jp(ONE_DATA_DIR, 'conv2d', run_timestamp)
        else:
            model_dir_path = jp(MANY_DATA_DIR, 'conv2d', run_timestamp)

        model_file_path = jp(model_dir_path, model_file_name)
        model_metadata_file_path = jp(model_dir_path, model_metadata_file_name)
        os.makedirs(model_dir_path, exist_ok=True)

        model.save_model(model_file_path)
        fu.serialize(meta_data, file_path=model_metadata_file_path, file_extension='json')

    @staticmethod
    def main(args):
        mode = str(args['mode'])
        max_num_nodes = (14 if args['num_nodes'] is None else int(args['num_nodes']))
        all_nodes = ['node-%02d' % i for i in range(1, max_num_nodes + 1)]
        overwrite = (False if args['overwrite'] is None else True if str(args['overwrite']).lower() == 'true'
        else False)
        test_nodes_id = ('2,13' if args['test_nodes'] is None else args['test_nodes'])
        test_nodes = ['node-%02d' % int(i) for i in test_nodes_id.split(',')]
        single_node = ('node-14' if args['single_node'] is None else 'node-%02d' % int(args['single_node']))
        # port_number = (9090 if args['port'] is None else int(args['port']))
        jobs = (-1 if args['jobs'] is None else int(args['jobs']))

        if mode == 'single':
            port_number = 9091
        else:
            port_number = 9080

        # Sanity pre-processing
        if single_node in test_nodes:
            test_nodes.remove(single_node)
            tmp = random.choice(all_nodes)
            test_nodes.append(tmp)
            all_nodes.remove(tmp)

        training_nodes = [i for i in all_nodes if i not in test_nodes]

        logger.info(f"Mode:          {mode}")
        logger.info(f"Overwrite:     {overwrite}")
        logger.info(f"All nodes:     {all_nodes}")
        logger.info(f"Train nodes:   {training_nodes}")
        logger.info(f"Test nodes:    {test_nodes}")
        logger.info(f"Single node:   {single_node}")
        logger.info(f"Max # nodes:   {max_num_nodes}")

        Modeling.train_node_model(training_nodes=training_nodes, test_nodes=test_nodes, single_node=single_node,
                                  mode=mode, overwrite=overwrite, port=port_number, n_jobs=jobs)


if __name__ == '__main__':
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("-o", "--overwrite", required=False, help="")
    argument_parser.add_argument("-tn", "--test_nodes", required=False, help="")
    argument_parser.add_argument("-sn", "--single_node", required=False, help="")
    argument_parser.add_argument("-n", "--num_nodes", required=False, help="")
    argument_parser.add_argument("-m", "--mode", required=False, help="single, many")
    argument_parser.add_argument("-j", "--jobs", required=False, help="integer")
    # argument_parser.add_argument("-p", "--port", required=False, help="9090, 9080")
    cmd_args = vars(argument_parser.parse_args())

    # Create parent directories if they do not exist
    os.makedirs(ONE_DATA_DIR, exist_ok=True)
    os.makedirs(MANY_DATA_DIR, exist_ok=True)
    os.makedirs(COMMON_DATA_DIR, exist_ok=True)
    Modeling.main(cmd_args)

# Many is better in:
# python -m pyutils.analysis.train_on_one_vs_many -n 13 -tn 5,6 -sn 9 -m many -o false
# python -m pyutils.analysis.train_on_one_vs_many -n 13 -tn 5,6 -sn 9 -m single -o false

# python -m pyutils.analysis.train_on_one_vs_many -n 14 -tn 12,8 -sn 9 -m many -o true
# python -m pyutils.analysis.train_on_one_vs_many -n 14 -tn 12,8 -sn 9 -m single -o true

# python -m pyutils.analysis.train_on_one_vs_many -n 14 -tn 6,11 -sn 12 -m single -o true
# python -m pyutils.analysis.train_on_one_vs_many -n 14 -tn 6,11 -sn 12 -m many -o true


# With dropping the replicas in the single node training data:
# python -m pyutils.analysis.train_on_one_vs_many -n 14 -tn 2,13 -sn 14 -m single -o true
# python -m pyutils.analysis.train_on_one_vs_many -n 14 -tn 2,13 -sn 14 -m many -o true
""" 
Using same data, with no duplicates dropping:
- almost identical results on one set of nodes.

Using the median of readings with droppping duplicates for single node: 
minor enhancement 77 vs 74 for the 5% error threshold

Using the average of readings with droppping duplicates for single node: 

"""
