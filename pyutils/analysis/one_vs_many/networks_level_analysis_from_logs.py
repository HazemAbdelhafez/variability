""" This file parses the log file of training job for one_vs_many experiment
at the network level. Why?
Due to a stupid timestamp issue, the models (for different SN) overwrite each other, I reran the experiments
# with a minute timestamp instead of an hour timestamp, for now I will parse results from the log.
"""

import json
from os.path import join as jp

import pandas as pd

from pyutils.characterization.networks.utils import VISION_NETWORKS, get_unified_benchmark_name, \
    get_print_benchmark_name
from pyutils.common.config import ALL_NODES
from pyutils.common.paths import DATA_DIR
from pyutils.common.strings import P_NETWORK
from pyutils.common.utils import FileUtils as FU

# Pandas display parameters
pd.set_option('display.max_rows', 8000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


class SavedModelData:
    def __init__(self):
        self.network = ''
        self.label = ''
        self.training_nodes = None
        self.testing_nodes = None
        self.training_data_shape = None
        self.testing_data_shape = None
        self.evaluation_results = dict()
        self.training_mode = ''

    def parse_network(self, line):
        line = line.split("['")[1].replace("']", "")
        self.network = get_unified_benchmark_name(line)

    def parse_training_nodes(self, line):
        line = line.split('         ')
        line = line[1].split(',')
        if type(line) is not list:
            line = [line]
        self.training_nodes = ['node-%02d' % int(i) for i in line]

    def parse_testing_nodes(self, line):
        line = line.split('         ')[1]
        self.testing_nodes = ['node-%02d' % int(line)]

    def parse_label(self, line):
        line = line.split('      ')[1]
        self.label = line

    def parse_results(self, line):
        line = line.replace("\'", "\"")
        line = dict(json.loads(line.split('[INFO] ')[1]))
        self.evaluation_results = line

    def parse_single_node(self, line):
        line = line.split('         ')[1]
        if line == 'None':
            return

        if self.training_mode != 'single':
            raise Exception("Invalid mode parsed from before")
        else:
            single_node = ['node-%02d' % int(line)]
            if len(single_node) != 1 or len(self.training_nodes) != 1:
                raise Exception("Invalid single node set from before")
            if single_node[0] != self.training_nodes[0]:
                raise Exception("Invalid single node set from before")

    def __str__(self):
        st = ""
        st += f"Network:        {self.network}\n"
        st += f"Mode:           {self.training_mode}\n"
        st += f"Train nodes:    {self.training_nodes}\n"
        st += f"Test nodes:     {self.testing_nodes}\n"
        st += f"Results:        {self.evaluation_results['5%']}\n"
        return st


def parse_networks_data(log_file_path, single_nodes, num_networks):
    lines = FU.deserialize(log_file_path)
    first_marker = 'Selected networks:'
    training_nodes_marker = 'Train nodes:'
    testing_nodes_marker = 'Test nodes:'
    label_marker = 'Modeling label:'
    end_marker = 'Evaluation results:'
    single_node_marker = "Single node: "
    found_a_network = False
    networks = list()
    nw = SavedModelData()
    i = 0
    while i < len(lines):
        line = lines[i]
        if not found_a_network and not line.__contains__(first_marker):
            pass
        elif line.__contains__(first_marker):
            nw = SavedModelData()
            nw.parse_network(line)
            found_a_network = True
        elif line.__contains__(training_nodes_marker):
            # TODO: there is an assumption here that we always run all networks modeling
            # even the multi-node ones.
            if len(networks) < num_networks:
                nw.parse_training_nodes(line)
                nw.training_mode = 'multiple'
            else:
                index = int((len(networks) - num_networks) / num_networks)
                nw.training_nodes = [single_nodes[index]]
                # nw.parse_training_nodes(line)
                nw.training_mode = 'single'
        elif line.__contains__(single_node_marker):
            nw.parse_single_node(line)
        elif line.__contains__(testing_nodes_marker):
            nw.parse_testing_nodes(line)

        elif line.__contains__(label_marker):
            nw.parse_label(line)
        elif line.__contains__(end_marker):
            line = lines[i + 1]
            nw.parse_results(line)
            found_a_network = False
            networks.append(nw)
        else:
            pass
        i += 1

    return networks


def calculate_improvement(sn_model_results, mn_model_results):
    results = dict()
    key = '5%'
    results['5%'] = 100 * (float(mn_model_results[key]) - float(sn_model_results[key])) / float(sn_model_results[key])
    key = '10%'
    results['10%'] = 100 * (float(mn_model_results[key]) - float(sn_model_results[key])) / float(sn_model_results[key])
    key = '15%'
    results['15%'] = 100 * (float(mn_model_results[key]) - float(sn_model_results[key])) / float(sn_model_results[key])
    key = '20%'
    results['20%'] = 100 * (float(mn_model_results[key]) - float(sn_model_results[key])) / float(sn_model_results[key])

    key = 'RMSE'
    results['RMSE'] = 100 * (float(sn_model_results[key]) - float(mn_model_results[key])) / float(sn_model_results[key])
    key = 'MAE'
    results['MAE'] = 100 * (float(sn_model_results[key]) - float(mn_model_results[key])) / float(sn_model_results[key])

    return results


def create_networks_improvement_table(single_node, log_file_path, single_nodes, num_networks):
    networks = VISION_NETWORKS
    networks_models_results = parse_networks_data(log_file_path, single_nodes, num_networks)

    # Categorize results
    mn_dict = dict()
    sn_dict = dict()
    for network_model_result in networks_models_results:
        if network_model_result.training_mode == 'single' and network_model_result.training_nodes[0] == single_node:
            sn_dict[network_model_result.network] = network_model_result.evaluation_results
        elif network_model_result.training_mode == 'multiple':
            mn_dict[network_model_result.network] = network_model_result.evaluation_results
        else:
            continue

    improvement_results = dict()
    for nw in networks:
        nw = get_unified_benchmark_name(nw)
        if nw in sn_dict.keys():
            improvement_results[get_print_benchmark_name(nw)] = calculate_improvement(sn_dict[nw], mn_dict[nw])

    # Create the DF
    df = pd.DataFrame(improvement_results).T.sort_index()
    df.drop(inplace=True, index=['VGG'])
    df[P_NETWORK] = df.index
    df.reset_index(inplace=True, drop=True)
    df = df[[P_NETWORK] + [i for i in df.columns if i != P_NETWORK]]

    # # Append mean row
    # mean_row = dict()
    # for col in df.columns:
    #     if col == P_NETWORK:
    #         mean_row[col] = 'Mean'
    #     else:
    #         mean_row[col] = df[col].mean()
    # df = df.append(pd.Series(mean_row), ignore_index=True)
    # df = df.round(2)

    # latex = df.to_latex(index=False)
    # latex = latex.replace("\\\\", "\\\\ \\hline")
    # print(latex)
    return df


def cat_and_summarize():
    """ Concat all networks then summarize min and max rows"""
    dfs = list()
    mode = 'power'
    if mode == 'power':
        test_nodes = ['node-02', 'node-13']
    else:
        test_nodes = ['node-02', 'node-13']

    for test_node in test_nodes:
        single_nodes = [i for i in ALL_NODES if i != test_node]
        if mode == 'power':
            log_file = 'one_vs_many_power_node_%02d_as_test.txt' % int(test_node.split('-')[1])
            num_networks = 7
        else:
            log_file = 'one_vs_many_node_%02d_as_test.txt' % int(test_node.split('-')[1])
            num_networks = 10
        log_file_path = jp(DATA_DIR, 'misc', log_file)
        for node in single_nodes:
            dfs.append(create_networks_improvement_table(node, log_file_path, single_nodes, num_networks))

    df = pd.concat(dfs)
    # df.drop(columns=['15%', '20%'], inplace=True)
    df.reset_index(inplace=True, drop=True)

    min_results = list()
    max_results = list()
    for nw, nw_data in df.groupby(by=['Network']):
        min_results.append(nw_data.min().to_dict())
        max_results.append(nw_data.max().to_dict())

    min_df = pd.DataFrame(min_results)
    max_df = pd.DataFrame(max_results)

    combined_df = pd.merge(min_df, max_df, on=['Network'])
    # Append mean row
    mean_row = dict()
    for col in combined_df.columns:
        if col == P_NETWORK:
            mean_row[col] = 'Mean'
        else:
            mean_row[col] = combined_df[col].mean()
    combined_df = combined_df.append(pd.Series(mean_row), ignore_index=True)
    combined_df = combined_df.round(2)

    for idx, row in combined_df.iterrows():
        print(format_latex_line(row))


def format_latex_line(df: pd.Series):
    line = ""
    line += df['Network'] + " & "
    line += f"{df['5%_x']} $\\rightarrow$ {df['5%_y']}  & "
    line += f"{df['10%_x']} $\\rightarrow$ {df['10%_y']}  & "
    line += f"{df['RMSE_x']} $\\rightarrow$ {df['RMSE_y']}  & "
    line += f"{df['MAE_x']} $\\rightarrow$ {df['MAE_y']}  \\\\hline"
    return line


cat_and_summarize()
