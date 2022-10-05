import json
import os

import pandas as pd

from pyutils.characterization.kernels.utils.checks import KernelsChecks
from pyutils.common.data_handlers.data_interface import DataHandler
from pyutils.common.strings import S_RUNTIME, S_POWER, S_KERNEL, S_LABEL
from pyutils.common.utils import FileUtils
from pyutils.modeling.helpers import ModelLoader, ModelSelectionFilter


class ModelStatisticsPrinter:
    @staticmethod
    def from_json(file_path: str):
        header = ['5%', '10%', '15%', '20%', 'RMSE (root mean squared error)', 'MAE (mean absolute error)']

        data = FileUtils.deserialize(file_path)
        dir_timestamp = file_path.split('_')[-1].rstrip('.json')

        print(f"Model ID:                   {data.get('timestamp', dir_timestamp)}")
        print(f"Kernel:                     {data[S_KERNEL]}")
        print(f"Label:                      {data[S_LABEL]}")
        print(f"Number of trees:            {data['num_boost_rounds']}")

        # Drop values we are not interested in
        data['model_params'].pop('eval_metric')
        data['model_params'].pop('nthread')
        data['model_params'].pop('objective')

        print(f"Best parameters:            {['number of trees'] + list(data['model_params'].keys())}")
        params_values = [data['num_boost_rounds']] + [str(i) for i in list(data['model_params'].values())]

        print(f"Best parameters values:     {','.join(params_values)}")
        summary_dict = dict()
        bcr = data['best_cv_results']
        summary_dict['CV Training'] = [bcr['train-percentage-with-error-less-than-5%-mean'],
                                       bcr['train-percentage-with-error-less-than-10%-mean'],
                                       bcr['train-percentage-with-error-less-than-15%-mean'],
                                       bcr['train-percentage-with-error-less-than-20%-mean'],
                                       bcr['train-real-rmse-mean'],
                                       bcr['train-real-mae-mean']]

        summary_dict['CV Evaluation'] = \
            [
                bcr['test-percentage-with-error-less-than-5%-mean'],
                bcr['test-percentage-with-error-less-than-10%-mean'],
                bcr['test-percentage-with-error-less-than-15%-mean'],
                bcr['test-percentage-with-error-less-than-20%-mean'],
                bcr['test-real-rmse-mean'],
                bcr['test-real-mae-mean']
            ]

        tr = data['training_results']
        summary_dict['Training'] = [
            tr['percentage-with-error-less-than-5%'],
            tr['percentage-with-error-less-than-10%'],
            tr['percentage-with-error-less-than-15%'],
            tr['percentage-with-error-less-than-20%'],
            tr['real-rmse'],
            tr['real-mae']
        ]
        er = data['evaluation_results']
        summary_dict['Evaluation'] = [
            er['percentage-with-error-less-than-5%'],
            er['percentage-with-error-less-than-10%'],
            er['percentage-with-error-less-than-15%'],
            er['percentage-with-error-less-than-20%'],
            er['real-rmse'],
            er['real-mae']
        ]

        df = pd.DataFrame()
        df = df.from_dict(summary_dict, orient='index', columns=header)
        df = df.astype(dtype=float)
        df = df.round(3)
        print()
        print(df)
        print()
        print("-->")
        # Custom printing of error stats
        for row in df.values:
            print(','.join([str(i) for i in row]))

        print()
        # Custom printing of meta data
        meta_data = dict()
        meta_data['Model ID'] = data.get('timestamp', dir_timestamp)
        meta_data['# Training data records'] = data['training_data_shape'][0]
        meta_data['# Evaluation data records'] = data['testing_data_shape'][0]
        meta_data['# Features'] = data['training_data_shape'][1]
        meta_data['# Training nodes'] = len(data['training_nodes'])
        meta_data['# Evaluation nodes'] = len(data['testing_nodes'])
        meta_data_df = pd.DataFrame(meta_data, index=[0]).astype(dtype=int)

        print(meta_data_df)
        print()
        print("-->")
        for row in meta_data_df.values:
            print(','.join([str(i) for i in row]))
        print()

    @staticmethod
    def print_summary(meta_files):

        header = ['5%', '10%', '15%', '20%', 'RMSE (root mean squared error)', 'MAE (mean absolute error)']
        print(','.join([S_KERNEL.capitalize()] + [i.split(' ')[0] for i in header]))
        for file_path in meta_files:
            with open(file_path) as file_obj:
                # print(file_path)
                data = json.load(file_obj)
                # Drop values we are not interested in
                data['model_params'].pop('eval_metric')
                data['model_params'].pop('nthread')
                data['model_params'].pop('objective')

                summary_dict = dict()
                bcr = data['best_cv_results']
                summary_dict['CV Evaluation'] = \
                    [
                        bcr['test-percentage-with-error-less-than-5%-mean'],
                        bcr['test-percentage-with-error-less-than-10%-mean'],
                        bcr['test-percentage-with-error-less-than-15%-mean'],
                        bcr['test-percentage-with-error-less-than-20%-mean'],
                        bcr['test-real-rmse-mean'],
                        bcr['test-real-mae-mean']
                    ]
                er = data['evaluation_results']
                summary_dict['Evaluation'] = [
                    er['percentage-with-error-less-than-5%'],
                    er['percentage-with-error-less-than-10%'],
                    er['percentage-with-error-less-than-15%'],
                    er['percentage-with-error-less-than-20%'],
                    er['real-rmse'],
                    er['real-mae']
                ]

                df = pd.DataFrame()
                df = df.from_dict(summary_dict, orient='index', columns=header)
                df = df.astype(dtype=float)
                df = df.round(3)

                # Custom printing of error stats
                row = df.loc['Evaluation', :]
                print(','.join([data[S_KERNEL].capitalize()] + ['{:,.3f}'.format(i) for i in row]))

    @staticmethod
    def main():
        for metric in [S_RUNTIME, S_POWER]:
            meta_files = list()
            label = DataHandler.get_metric_label(metric)
            selection_filter = ModelSelectionFilter(n_from_last=2)
            print(f"{label} model summary")
            for kernel in KernelsChecks.get_supported_kernels():
                meta_file_path = ModelLoader.get_trained_model_meta_path(kernel=kernel, label=label,
                                                                         selection_filter=selection_filter)
                meta_files.append(meta_file_path)
                print(meta_file_path)
            for i in meta_files:
                if not os.path.exists(i):
                    print(f"{i} does not exist")
            ModelStatisticsPrinter.print_summary(meta_files)
            print()


if __name__ == '__main__':
    ModelStatisticsPrinter.main()
