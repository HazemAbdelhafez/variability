from os.path import join as jp

import pandas as pd

from pyutils.analysis.one_vs_many.config import ONE_VS_MANY_DIR
from pyutils.common.strings import S_RUNTIME_MS, S_CONV
from pyutils.common.utils import GlobalLogger, FileUtils
from pyutils.modeling.helpers import ModelLoader

logger = GlobalLogger().get_logger()


class OneVersusManyAnalysis:
    @staticmethod
    def get_single_node_model(node, timestamp=''):
        parent_model_dir = jp(ONE_VS_MANY_DIR, node, 'one')
        return ModelLoader.get_trained_model_meta_path(kernel=S_CONV, timestamp=timestamp, label=S_RUNTIME_MS,
                                                       parent_dir=parent_model_dir)

    @staticmethod
    def get_multiple_nodes_model(node, timestamp=''):
        parent_model_dir = jp(ONE_VS_MANY_DIR, node, 'many')
        return ModelLoader.get_trained_model_meta_path(kernel=S_CONV, timestamp=timestamp, label=S_RUNTIME_MS,
                                                       parent_dir=parent_model_dir)

    @staticmethod
    def print_summary(single_node_model, many_nodes_model):
        header = ['5%', '10%', '15%', '20%', 'RMSE (root mean squared error)', 'MAE (mean absolute error)']
        # print(','.join([S_KERNEL.capitalize()] + [i.split(' ')[0] for i in header]))

        summary = list()
        single_node_eval = dict()
        er = single_node_model['evaluation_results']
        single_node_eval['Model'] = 'Single node'
        single_node_eval['5%'] = er['percentage-with-error-less-than-5%']
        single_node_eval['10%'] = er['percentage-with-error-less-than-10%']
        single_node_eval['15%'] = er['percentage-with-error-less-than-15%']
        single_node_eval['20%'] = er['percentage-with-error-less-than-20%']
        single_node_eval['RMSE (ms)'] = er['real-rmse']
        single_node_eval['MAE (ms)'] = er['real-mae']

        many_nodes_eval = dict()
        er = many_nodes_model['evaluation_results']
        many_nodes_eval['Model'] = 'Multiple nodes'
        many_nodes_eval['5%'] = er['percentage-with-error-less-than-5%']
        many_nodes_eval['10%'] = er['percentage-with-error-less-than-10%']
        many_nodes_eval['15%'] = er['percentage-with-error-less-than-15%']
        many_nodes_eval['20%'] = er['percentage-with-error-less-than-20%']
        many_nodes_eval['RMSE (ms)'] = er['real-rmse']
        many_nodes_eval['MAE (ms)'] = er['real-mae']

        improvement_dict = dict()
        improvement_dict['Model'] = 'Improvement'
        improvement_dict['5%'] = f"{float(many_nodes_eval['5%']) - float(single_node_eval['5%'])} "
        improvement_dict['10%'] = f"{float(many_nodes_eval['10%']) - float(single_node_eval['10%'])} "
        improvement_dict['15%'] = f"{float(many_nodes_eval['15%']) - float(single_node_eval['15%'])} "
        improvement_dict['20%'] = f"{float(many_nodes_eval['20%']) - float(single_node_eval['20%'])} "
        improvement_dict[
            'RMSE (ms)'] = f"{100 * (float(single_node_eval['RMSE (ms)']) - float(many_nodes_eval['RMSE (ms)'])) / float(single_node_eval['RMSE (ms)'])} "
        improvement_dict[
            'MAE (ms)'] = f"{100 * (float(single_node_eval['MAE (ms)']) - float(many_nodes_eval['MAE (ms)'])) / float(single_node_eval['MAE (ms)'])} "

        summary.append(single_node_eval)
        summary.append(many_nodes_eval)
        summary.append(improvement_dict)

        df = pd.DataFrame(summary)
        numeric_cols = [i for i in df.columns if i != 'Model']
        df[numeric_cols] = df[numeric_cols].astype(dtype=float)
        df[numeric_cols] = df[numeric_cols].round(3)

        print(df.to_latex(index=False))

    @staticmethod
    def main():
        single_meta_file_path = OneVersusManyAnalysis.get_single_node_model(node='node290')
        multiple_meta_file_path = OneVersusManyAnalysis.get_multiple_nodes_model(node='node290')
        logger.info(f"Single node meta   : {single_meta_file_path}")
        logger.info(f"Multiple nodes meta: {multiple_meta_file_path}")
        # ModelStatisticsPrinter.print_summary([single_meta_file_path, multiple_meta_file_path])
        single_node_model_data = FileUtils.deserialize(single_meta_file_path)
        many_nodes_model_data = FileUtils.deserialize(multiple_meta_file_path)
        OneVersusManyAnalysis.print_summary(single_node_model_data, many_nodes_model_data)


if __name__ == '__main__':
    OneVersusManyAnalysis.main()
