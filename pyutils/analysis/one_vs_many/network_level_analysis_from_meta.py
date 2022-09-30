# Pandas display parameters
import glob

import pandas as pd

from pyutils.analysis.common.error_metrics import MetricPrettyName
from pyutils.characterization.networks.utils import get_unified_benchmark_name, get_print_benchmark_name
from pyutils.common.paths import ONE_VS_MANY_MODELS_DIR
from pyutils.common.strings import S_NETWORK, S_RUNTIME_MS, S_AVG_PWR_W, S_LABEL, S_TESTING_NODES, S_TRAINING_NODES
from pyutils.common.utils import FileUtils
from pyutils.common.utils import GlobalLogger

logger = GlobalLogger().get_logger()

S_MODE = "mode"

pd.set_option('display.max_rows', 8000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


class SavedModelData:
    def __init__(self, fc):
        self.network = ''
        self.label = ''
        self.training_nodes = None
        self.testing_nodes = None
        self.training_data_shape = None
        self.testing_data_shape = None
        self.evaluation_results = dict()
        self.training_mode = ''
        self.parse_all(fc)

    def parse_network(self, fc: dict):
        self.network = get_unified_benchmark_name(fc.get(S_NETWORK))

    def parse_training_nodes(self, fc: dict):
        self.training_nodes = fc.get(S_TRAINING_NODES)
        if len(self.training_nodes) == 1:
            self.training_mode = 'single'
        else:
            self.training_mode = 'multiple'

    def parse_testing_nodes(self, fc: dict):
        self.testing_nodes = fc.get(S_TESTING_NODES)

    def parse_label(self, fc: dict):
        self.label = fc.get(S_LABEL)

    def parse_evaluation(self, fc: dict):
        self.evaluation_results = MetricPrettyName.get_pretty_evaluation_metrics(fc.get("evaluation_results"))

    def __str__(self):
        st = ""
        st += f"Network:        {self.network}\n"
        st += f"Mode:           {self.training_mode}\n"
        st += f"Label:          {self.label}\n"
        st += f"Train nodes:    {self.training_nodes}\n"
        st += f"Test nodes:     {self.testing_nodes}\n"
        st += f"Results:        {self.evaluation_results['5%']}\n"
        return st

    def parse_all(self, fc):
        self.parse_network(fc)
        self.parse_label(fc)
        self.parse_training_nodes(fc)
        self.parse_testing_nodes(fc)
        self.parse_evaluation(fc)

    def to_dict(self):
        d = dict()
        d[S_NETWORK] = self.network
        d[S_LABEL] = self.label
        d[S_TESTING_NODES] = ','.join([str(int(i.split('-')[1])) for i in self.testing_nodes])
        d[S_TRAINING_NODES] = ','.join([str(int(i.split('-')[1])) for i in self.training_nodes])
        d[S_MODE] = self.training_mode
        return {**d, **self.evaluation_results}


class NetworkLevelOneVsManyAnalysis:
    @staticmethod
    def parse_meta_files():
        meta_files = glob.glob(f'{ONE_VS_MANY_MODELS_DIR}/**/*.json', recursive=True)
        all_results = list()

        for fp in meta_files:
            fc = FileUtils.deserialize(fp)
            saved_model_results = SavedModelData(fc)
            all_results.append(saved_model_results.to_dict())

        # Dataframe
        df = pd.DataFrame(all_results)
        df.sort_values(by=[S_NETWORK, S_LABEL, S_TESTING_NODES, S_TRAINING_NODES], inplace=True, ignore_index=True)
        df.drop_duplicates(inplace=True, ignore_index=True)
        return df

    @staticmethod
    def calculate_improvement(sn_model_results, mn_model_results):
        results = dict()
        key = '5%'
        results['5%'] = 100 * (float(mn_model_results[key]) - float(sn_model_results[key])) / float(
            sn_model_results[key])
        key = '10%'
        results['10%'] = 100 * (float(mn_model_results[key]) - float(sn_model_results[key])) / float(
            sn_model_results[key])

        # Beyond 10% is a very small difference that we neglect
        # key = '15%'
        # results['15%'] = 100 * (float(mn_model_results[key]) - float(sn_model_results[key])) / float(
        #     sn_model_results[key])
        # key = '20%'
        # results['20%'] = 100 * (float(mn_model_results[key]) - float(sn_model_results[key])) / float(
        #     sn_model_results[key])
        key = 'RMSE'
        results['RMSE'] = 100 * (float(sn_model_results[key]) - float(mn_model_results[key])) / float(
            sn_model_results[key])
        key = 'MAE'
        results['MAE'] = 100 * (float(sn_model_results[key]) - float(mn_model_results[key])) / float(
            sn_model_results[key])

        return results

    @staticmethod
    def process_runtime_models():
        return NetworkLevelOneVsManyAnalysis._process_label_models(label=S_RUNTIME_MS)

    @staticmethod
    def process_power_models():
        return NetworkLevelOneVsManyAnalysis._process_label_models(label=S_AVG_PWR_W)

    @staticmethod
    def _process_label_models(label):
        df = NetworkLevelOneVsManyAnalysis.parse_meta_files()
        df.drop(index=df[df[S_LABEL] != label].index, inplace=True)
        df.reset_index(inplace=True, drop=True)
        groups = df.groupby(by=[S_NETWORK, S_TESTING_NODES])

        improvement_summary = list()
        for nw_and_test_node, nw_data in groups:
            # Multiple nodes mode data
            multi_node_data = nw_data[nw_data[S_MODE] == 'multiple'].copy()

            # Single node mode data
            single_node_data = nw_data[nw_data[S_MODE] == 'single'].copy()

            # There should be only one multi-node experiment. Hence it is safe to index the first without checking
            multi_node_data = multi_node_data.to_dict(orient='records')
            if len(multi_node_data) != 1:
                logger.warning("More than multi-node experiment detected. Picking first.")
            multi_node_data = multi_node_data[0]

            single_node_data = single_node_data.to_dict(orient='records')

            for sn in single_node_data:
                improvement_data = NetworkLevelOneVsManyAnalysis.calculate_improvement(sn, multi_node_data)
                improvement_data[S_NETWORK] = get_print_benchmark_name(sn[S_NETWORK])
                improvement_data[S_TESTING_NODES] = sn[S_TESTING_NODES]
                improvement_data[S_TRAINING_NODES] = sn[S_TRAINING_NODES]
                improvement_summary.append(improvement_data)

        tmp = pd.DataFrame(improvement_summary)
        logger.info(f"Label: {label} - Size: {tmp.shape[0]}")  # Size should be = #_nws * #_test_nodes * (#_nodes-1)
        return tmp

    @staticmethod
    def format_latex_line(df: pd.Series):
        line = ""
        if df[S_NETWORK] != 'Mean':
            line += df[S_NETWORK] + " & "
            line += f"{df['5%_x']} $\\rightarrow$ {df['5%_y']}  & "
            line += f"{df['10%_x']} $\\rightarrow$ {df['10%_y']}  & "
            line += f"{df['RMSE_x']} $\\rightarrow$ {df['RMSE_y']}  & "
            line += f"{df['MAE_x']} $\\rightarrow$ {df['MAE_y']}  \\\\ \\hline"
        else:
            line += "\\textbf{" + df[S_NETWORK] + "} & "
            line += "\\textbf{" + f"{df['5%_x']}" + "} " + "$\\rightarrow$" + " \\textbf{" + f"{df['5%_y']}" + "} & "
            line += "\\textbf{" + f"{df['10%_x']}" + "} " + "$\\rightarrow$" + " \\textbf{" + f"{df['10%_y']}" + "} & "
            line += "\\textbf{" + f"{df['RMSE_x']}" + "} " + "$\\rightarrow$" + \
                    " \\textbf{" + f"{df['RMSE_y']}" + "} & "
            line += "\\textbf{" + f"{df['MAE_x']}" + "} " + "$\\rightarrow$" + \
                    " \\textbf{" + f"{df['MAE_y']}" + "} \\\\ \\hline"
        return line

    @staticmethod
    def create_and_print_table(df):
        df.drop(inplace=True, index=df[df[S_NETWORK] == 'VGG'].index)
        df.reset_index(inplace=True, drop=True)
        df = df[[S_NETWORK] + [i for i in df.columns if i != S_NETWORK]].copy()

        # Drop testing and training nodes
        df.drop(columns=[S_TESTING_NODES, S_TRAINING_NODES], inplace=True)

        min_results = list()
        max_results = list()
        for nw, nw_data in df.groupby(by=[S_NETWORK]):
            min_results.append(nw_data.min().to_dict())
            max_results.append(nw_data.max().to_dict())

        min_df = pd.DataFrame(min_results)
        max_df = pd.DataFrame(max_results)

        combined_df = pd.merge(min_df, max_df, on=[S_NETWORK])

        # Append mean row
        mean_row = dict()
        for col in combined_df.columns:
            if col == S_NETWORK:
                mean_row[col] = 'Mean'
            else:
                mean_row[col] = combined_df[col].mean()
        combined_df = combined_df.append(pd.Series(mean_row), ignore_index=True)
        combined_df = combined_df.round(1)

        for idx, row in combined_df.iterrows():
            print(NetworkLevelOneVsManyAnalysis.format_latex_line(row))

    @staticmethod
    def main():
        d = NetworkLevelOneVsManyAnalysis.process_runtime_models()
        NetworkLevelOneVsManyAnalysis.create_and_print_table(d)
        print()
        d = NetworkLevelOneVsManyAnalysis.process_power_models()
        NetworkLevelOneVsManyAnalysis.create_and_print_table(d)


if __name__ == '__main__':
    NetworkLevelOneVsManyAnalysis.main()
