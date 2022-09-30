import sys
from os.path import join as join_path
from random import choice as pick

import numpy as np
import pandas as pd
import xgboost as xgb
from pyutils.modeling.training_models import BatchNormModel

from pyutils.analysis.common.data_transformers import Transformers
from pyutils.analysis.common.error_metrics import Metrics
from pyutils.characterization.kernels.utils.loaders import KernelsLoaders
from pyutils.common.config import ALL_LABELS
from pyutils.common.strings import S_CONV, S_BATCHNORM, S_MATMUL, S_MAXPOOL, S_RELU, \
    S_ADAPTIVEPOOL
from pyutils.common.strings import S_RUNTIME_MS, S_CPU_FREQ, S_GPU_FREQ, S_MEMORY_FREQ, S_AVG_PWR_W
from pyutils.hosts.agx import DVFSHelpers
from pyutils.modeling.helpers import TrainingHelpers, ModelLoader

FINAL_MODELS = {
    S_CONV: {
        S_AVG_PWR_W: 'generated/saved-models/conv2d/node260/0724122020/xgb_avg_power_0724122020.model',
        S_RUNTIME_MS: 'generated/saved-models/conv2d/node260/1704022021/xgb_runtime_ms_1704022021.model',
    },
    S_BATCHNORM: {
        S_AVG_PWR_W: 'generated/saved-models/batchnorm2d/node260/0924122020/xgb_avg_power_0924122020.model',
        # S_RUNTIME_MS: 'generated/saved-models/batchnorm2d/node260/1326122020/xgb_runtime_ms_1326122020.model',
        S_RUNTIME_MS: 'generated/saved-models/batchnorm2d/inspiron/2001022021/xgb_runtime_ms_2001022021.model'
    },
    S_MATMUL: {
        S_AVG_PWR_W: 'generated/saved-models/matmuladd/node260/1024122020/xgb_avg_power_1024122020.model',
        S_RUNTIME_MS: 'generated/saved-models/matmuladd/node260/1326122020/xgb_runtime_ms_1326122020.model'
    },
    S_MAXPOOL: {
        S_AVG_PWR_W: 'generated/saved-models/maxpool2d/node260/1224122020/xgb_avg_power_1224122020.model',
        S_RUNTIME_MS: 'generated/saved-models/maxpool2d/node260/1526122020/xgb_runtime_ms_1526122020.model'
    },
    S_ADAPTIVEPOOL: {
        S_AVG_PWR_W: '',
        S_RUNTIME_MS: ''
    },
    S_RELU: {
        S_AVG_PWR_W: 'generated/saved-models/relu/node260/1629122020/xgb_avg_power_1629122020.model',
        S_RUNTIME_MS: 'generated/saved-models/relu/node260/1328122020/xgb_runtime_ms_1328122020.model'
    }
}


class Helpers:
    @staticmethod
    def get_intersections(df1: pd.DataFrame, df2: pd.DataFrame, non_label_columns):
        # Returns the intersecting rows between df1 and df2 based on the values in columns
        int_df = pd.merge(df1, df2, how='inner', on=non_label_columns)
        return int_df

    @staticmethod
    def find_records_with_common_parameters(kernel, nodes):

        model = TrainingHelpers.load_kernel_model(kernel)
        if model is None:
            return

        print("Kernel: ", kernel)
        df_dict = model.read_characterization_data_summary(nodes=nodes, keep_labels=[S_RUNTIME_MS],
                                                           overwrite_summary=False)
        non_label_columns = [i for i in list(df_dict.values())[0].columns if i not in ALL_LABELS]

        keys = list(df_dict.keys())
        n = len(keys)
        intersection_dict = dict()
        for i in range(n - 1):
            for j in range(i + 1, n):
                # Merge method returns the other columns post fixed with x and y
                intersection_dict[f"{keys[i]}_{keys[j]}"] = \
                    Helpers.get_intersections(df_dict[keys[i]], df_dict[keys[j]], non_label_columns)

        # print("Number of intersecting records stats: ")
        # for key, val in intersection_dict.items():
        #     print(f"{key} : {val.head(1)}")
        return intersection_dict

    @staticmethod
    def simulate_ideal_model_training(kernel):
        train_nodes = ['node-%02d' % i for i in range(1, 7)]
        test_nodes = ['node-%02d' % i for i in range(7, 14)]
        model = TrainingHelpers.load_kernel_model(kernel)
        if model is None:
            return
        nodes = train_nodes + test_nodes
        df_dict = model.read_characterization_data_summary(nodes=nodes, keep_labels=[S_RUNTIME_MS],
                                                           overwrite_summary=False)
        non_label_columns = [i for i in list(df_dict.values())[0].columns if i not in ALL_LABELS]

        df = pd.DataFrame()
        first_time = True
        for key in df_dict.keys():
            df_dict[key][f'{key}_{S_RUNTIME_MS}'] = Transformers.inverse_log_transform(df_dict[key][S_RUNTIME_MS])
            df_dict[key].drop(columns=[S_RUNTIME_MS], inplace=True)
            if first_time:
                df = df_dict[key]
                first_time = False
            else:
                df = pd.merge(df, df_dict[key], how='inner', on=non_label_columns)

        df.drop(columns=[i for i in non_label_columns], inplace=True)

        for idx, row in df.iterrows():
            # Test nodes best predictions
            for test_node in test_nodes:
                error_values = list()
                runtime_values = list()
                for train_node in train_nodes:
                    error_values.append(abs(row[f'{test_node}_{S_RUNTIME_MS}'] - row[f'{train_node}_{S_RUNTIME_MS}']))
                    runtime_values.append(row[f'{train_node}_{S_RUNTIME_MS}'])
                avg_runtime = np.mean(runtime_values)
                error_values = np.array(error_values)
                best_predictor_node_id = np.argmin(error_values) + 1
                df.loc[idx, f'{test_node}_{S_RUNTIME_MS}_pred'] = \
                    avg_runtime
                # row[f'node-%02d_{S_RUNTIME_MS}' % best_predictor_node_id]

        # Report rmse values
        rmse_values = list()
        for test_node in test_nodes:
            rmse = Metrics.rmse(df[f'{test_node}_{S_RUNTIME_MS}_pred'], df[f'{test_node}_{S_RUNTIME_MS}'])
            rmse_values.append(rmse)

        # N-nodes model
        print(rmse_values)

        # 1-node model
        # Picked oracle node = node-03
        rmse_values = list()
        for test_node in test_nodes:
            rmse = Metrics.rmse(df[f'{test_node}_{S_RUNTIME_MS}'], df[f'node-03_{S_RUNTIME_MS}'])
            rmse_values.append(rmse)
        print(rmse_values)

    @staticmethod
    def report_stats(kernel, intersection_dict: dict, predict: bool = False):
        labels_x = [f"{i}_x" for i in ALL_LABELS]
        labels_y = [f"{i}_y" for i in ALL_LABELS]

        # Apply inverse transformation
        for key, df in intersection_dict.items():
            for label in labels_x:
                if label in df.columns:
                    df[label] = Transformers.inverse_log_transform(df[label])

            for label in labels_y:
                if label in df.columns:
                    df[label] = Transformers.inverse_log_transform(df[label])

        # Remove the labels that don't exist in the dataframe columns
        for key, df in intersection_dict.items():
            labels_x = [i for i in labels_x if i in df.columns]
            labels_y = [i for i in labels_y if i in df.columns]

        # Create error report dataframe
        df_overall = pd.DataFrame()
        for key, df in intersection_dict.items():
            df_summary = pd.DataFrame()
            for x, y in zip(labels_x, labels_y):
                label_name = '_'.join(x.split('_')[:-1])
                df_summary[f'{label_name}_x'] = df[x]
                df_summary[f'{label_name}_y'] = df[y]
                df_summary[f'{label_name}_percentage_error'] = Metrics.relative_error_percentage(df[x], df[y])
                df_summary[f'{label_name}_abs_error'] = Metrics.abs_error(df[x], df[y])
                if key.split('_')[0] == 'node-03' and key.split('_')[1] == 'node-07':
                    print(f"RMSE of {key} for {label_name}: {Metrics.rmse(df[x], df[y])}")
                    print(df_summary.shape)
                    return
                    # print()

                if predict:
                    # Drop labels, and prepare prediction inputs
                    model_inputs_df = df.drop(columns=labels_x + labels_y)
                    model_inputs_dm = xgb.DMatrix(model_inputs_df)
                    model = ModelLoader.load_trained_model(kernel, label_name)
                    prediction = model.predict(model_inputs_dm)
                    df_summary[f'{label_name}_prediction'] = Transformers.inverse_log_transform(prediction)

            # save the pdf to csv for later
            output_file_dir = join_path(CROSS_NODE_ANALYSIS, 'results', kernel)
            os.makedirs(output_file_dir, exist_ok=True)

            output_file_path = join_path(output_file_dir, f'{key}.csv')
            df_summary.to_csv(output_file_path, index=False)

            # print(f"Error summary of {key}")
            # print(df_summary.describe())
            # print()
            df_overall = df_overall.append(df_summary)
        # print()

    @staticmethod
    def reload_results_and_report_stats(kernel=None, prediction=False, report_power=False):
        results_dir = join_path(CROSS_NODE_ANALYSIS, 'results')
        if kernel is None:
            results_dir_contents = os.listdir(results_dir)
        else:
            results_dir_contents = [kernel]

        for kernel_dir_name in results_dir_contents:
            print(f"Kernel: {kernel_dir_name}")
            kernel_results_dir_path = join_path(results_dir, kernel_dir_name)

            # Extract the number of nodes to construct the result matrix
            files = os.listdir(kernel_results_dir_path)
            nodes = list()
            for file_name in files:
                node_x, node_y = file_name.replace('.csv', '').split('_')
                if node_x not in nodes:
                    nodes.append(node_x)
                if node_y not in nodes:
                    nodes.append(node_y)
            nodes.sort()

            labels_x = [f"{i}_x" for i in ALL_LABELS]
            labels_y = [f"{i}_y" for i in ALL_LABELS]

            # Construct the result matrix based on the nodes
            runtime_results_df = pd.DataFrame(index=nodes + ['model'], columns=nodes)
            runtime_results_df = runtime_results_df.fillna(0.0)
            power_results_df = pd.DataFrame(index=nodes + ['model'], columns=nodes)
            power_results_df = power_results_df.fillna(0.0)
            for csv_file in os.listdir(kernel_results_dir_path):
                df = pd.read_csv(join_path(kernel_results_dir_path, csv_file))
                node_x, node_y = csv_file.replace('.csv', '').split('_')
                # Remove the labels that don't exist in the dataframe columns
                labels_x = [i for i in labels_x if i in df.columns]
                labels_y = [i for i in labels_y if i in df.columns]

                # Collect and print results for runtime
                for x, y in zip(labels_x, labels_y):
                    label_name = '_'.join(x.split('_')[:-1])
                    # print(f"{node_x} and {node_y} RMSE for {label_name}: {Metrics.rmse(df[x], df[y])}")
                    # print(f"Predictor and {node_y} RMSE for {label_name}: "
                    #       f"{Metrics.rmse(df[f'{label_name}_prediction'], df[y])}")
                    # print()
                    df[f'{label_name}_pred_percentage_error'] = Metrics.relative_error_percentage(
                        df[f'{label_name}_prediction'], df[y])
                    df[f'{label_name}_pred_abs_error'] = Metrics.abs_error(df[f'{label_name}_prediction'], df[y])
                    if label_name == S_RUNTIME_MS:
                        runtime_results_df.at[node_x, node_y] = Metrics.rmse(df[x], df[y])
                        runtime_results_df.at[node_y, node_x] = Metrics.rmse(df[x], df[y])
                        runtime_results_df.at['model', node_x] = Metrics.rmse(df[f'{label_name}_prediction'], df[x])
                        runtime_results_df.at['model', node_y] = Metrics.rmse(df[f'{label_name}_prediction'], df[y])
                    elif label_name == S_AVG_PWR_W:
                        power_results_df.at[node_x, node_y] = Metrics.rmse(df[x], df[y])
                        power_results_df.at['model', node_x] = Metrics.rmse(df[f'{label_name}_prediction'], df[x])
                        power_results_df.at['model', node_y] = Metrics.rmse(df[f'{label_name}_prediction'], df[y])
                    else:
                        continue

            runtime_results_df = runtime_results_df.round(3)
            power_results_df = power_results_df.round(3)
            print("Runtime results:")
            print(runtime_results_df)
            print("-->")
            print(','.join(list(runtime_results_df.columns)))
            for row in runtime_results_df.values:
                row_items = list()
                for i in row:
                    # row_items.append(str(i) if i != -1 else 'NA')
                    row_items.append(str(i))
                print(','.join([i for i in row_items]))
            if report_power:
                print()
                print("Power results:")
                print(power_results_df)
                print("-->")
                for row in power_results_df.values:
                    row_items = list()
                    for i in row:
                        row_items.append(str(i) if i != -1 else 'x')
                    print(','.join([i for i in row_items]))
                print()

    @staticmethod
    def manual_model_inspection():
        # True data loading
        record = {"runtime_ms": 12355.677162, "iterations": 50, "valid_iterations": 50, "metric": "time",
                  "dvfs_config": {"cpu_freq": 806400, "gpu_freq": 114750000, "memory_freq": 2133000000},
                  "kernel_parameters": {"n": 1, "c": 2432, "h": 182, "w": 231, "num_features": 2432},
                  "time_per_run_ms": [154.66085815429688, 154.5861053466797, 154.6096649169922, 154.57594299316406,
                                      154.6033935546875, 154.58099365234375, 154.5707550048828, 154.5963592529297,
                                      154.58624267578125, 154.6024627685547, 154.5482177734375, 154.5891876220703,
                                      154.58714294433594, 154.56675720214844, 154.59939575195312, 154.63340759277344,
                                      154.5850830078125, 154.54319763183594, 154.5503387451172, 154.59756469726562,
                                      154.59426879882812, 154.5861053466797, 154.58303833007812, 154.5656280517578,
                                      154.56886291503906, 154.55130004882812, 154.58303833007812, 154.56448364257812,
                                      154.56871032714844, 154.59030151367188, 154.60348510742188, 154.57583618164062,
                                      154.546875, 154.54412841796875, 154.56460571289062, 154.5840301513672,
                                      154.61375427246094, 154.588134765625, 154.56871032714844, 154.56655883789062,
                                      154.6229705810547, 154.6219482421875, 154.58303833007812, 154.57379150390625,
                                      154.64732360839844, 154.59750366210938, 154.63612365722656, 154.57586669921875,
                                      154.56764221191406, 154.56150817871094]}
        record = {"runtime_ms": 6183.464503, "iterations": 50, "valid_iterations": 50, "metric": "time",
                  "dvfs_config": {"cpu_freq": 2265600, "gpu_freq": 1377000000, "memory_freq": 2133000000},
                  "kernel_parameters": {"n": 1, "c": 58, "h": 28, "w": 28, "num_features": 58},
                  "time_per_run_ms": [3.859488010406494, 2.4043519496917725, 2.4248640537261963, 2.4022719860076904,
                                      2.40230393409729, 2.409503936767578, 2.414560079574585, 2.408479928970337,
                                      2.4207680225372314, 2.406399965286255, 2.3941121101379395, 2.416640043258667,
                                      2.434015989303589, 2.4104959964752197, 2.4022719860076904, 2.4114880561828613,
                                      2.407423973083496, 2.4064319133758545, 2.3992319107055664, 2.4197120666503906,
                                      2.4054079055786133, 2.4145920276641846, 2.417664051055908, 2.401279926300049,
                                      2.4084479808807373, 2.4156479835510254, 2.4043519496917725, 2.4084160327911377,
                                      2.397183895111084, 2.407423973083496, 2.401279926300049, 2.404320001602173,
                                      2.417664051055908, 2.425856113433838, 2.3930881023406982, 2.396127939224243,
                                      2.4104959964752197, 2.416640043258667, 2.412544012069702, 2.420736074447632,
                                      2.4094719886779785, 2.4176321029663086, 2.407423973083496, 2.4104959964752197,
                                      2.418720006942749, 2.397183895111084, 2.4145920276641846, 2.3961598873138428,
                                      2.4002559185028076, 2.4135680198669434]}
        record = {"runtime_ms": 507.426762, "iterations": 50, "metric": "time",
                  "dvfs_config": {"cpu_freq": 345600, "gpu_freq": 1377000000, "memory_freq": 1065600000},
                  "kernel_parameters": {"n": 1, "c": 48, "h": 70, "w": 70, "in_channels": 48, "out_channels": 944,
                                        "kernel_h": 7, "kernel_w": 7, "bias": 1, "stride_h": 2, "stride_w": 2,
                                        "padding_h": 0, "padding_w": 0, "dilation_h": 1, "dilation_w": 1, "groups": 1,
                                        "padding_mode": "zeros"},
                  "time_per_run_ms": [7.454527854919434, 6.83622407913208, 6.832736015319824, 6.737919807434082,
                                      6.758399963378906, 6.540800094604492, 6.7491841316223145, 6.725152015686035,
                                      6.775519847869873, 6.594592094421387, 6.540095806121826, 6.724800109863281,
                                      6.570623874664307, 6.716512203216553, 6.713888168334961, 6.470240116119385,
                                      6.777535915374756, 6.7330241203308105, 6.5610880851745605, 6.6263041496276855,
                                      6.348800182342529, 6.53004789352417, 6.599679946899414, 6.6754560470581055,
                                      6.634079933166504, 6.404767990112305, 6.784319877624512, 6.664927959442139,
                                      6.63756799697876, 6.514368057250977, 6.546527862548828, 6.753280162811279,
                                      6.7505598068237305, 6.4008002281188965, 6.446944236755371, 6.470016002655029,
                                      6.542943954467773, 6.699999809265137, 6.691775798797607, 6.45417594909668,
                                      6.399648189544678, 6.3600640296936035, 6.437952041625977, 6.564671993255615,
                                      6.468448162078857, 6.3466877937316895, 6.461120128631592, 6.455552101135254,
                                      6.5049920082092285, 6.464320182800293]}
        # record = \
        #         {"runtime_ms": 7646.7871, "iterations": 50, "valid_iterations": 50, "metric": "time", "dvfs_config": {"cpu_freq": 960000, "gpu_freq": 318750000, "memory_freq": 408000000}, "kernel_parameters": {"n": 1, "c": 1744, "h": 245, "w": 245, "in_channels": 1744, "out_channels": 8, "kernel_h": 3, "kernel_w": 3, "bias": 1, "stride_h": 2, "stride_w": 2, "padding_h": 3, "padding_w": 3, "dilation_h": 1, "dilation_w": 1, "groups": 1, "padding_mode": "zeros"}, "time_per_run_ms": [412.3023681640625, 141.9356231689453, 141.95712280273438, 142.09738159179688, 141.918212890625, 142.0400390625, 142.1609649658203, 141.8802947998047, 141.8660125732422, 141.9734649658203, 141.99913024902344, 141.98272705078125, 141.92025756835938, 141.87826538085938, 141.9990997314453, 141.90797424316406, 141.95510864257812, 141.86492919921875, 141.876220703125, 141.94483947753906, 142.00729370117188, 141.94586181640625, 141.9990997314453, 141.85667419433594, 141.98374938964844, 141.8670654296875, 141.99293518066406, 142.0298309326172, 141.92335510253906, 141.94076538085938, 142.06973266601562, 142.02879333496094, 141.97760009765625, 141.83631896972656, 141.99090576171875, 141.88543701171875, 142.03187561035156, 141.9376678466797, 141.9110107421875, 142.10662841796875, 141.90797424316406, 141.98681640625, 141.91001892089844, 141.88441467285156, 142.03289794921875, 142.05133056640625, 141.87930297851562, 141.96424865722656, 141.90386962890625, 141.97760009765625]}
        record = {"runtime_ms": 1789.107332, "iterations": 50, "valid_iterations": 50, "metric": "time",
                  "dvfs_config": {"cpu_freq": 2265600, "gpu_freq": 420750000, "memory_freq": 204000000},
                  "kernel_parameters": {"n": 1, "c": 448, "h": 224, "w": 224, "in_channels": 448, "out_channels": 1144,
                                        "kernel_h": 1, "kernel_w": 1, "bias": 1, "stride_h": 4, "stride_w": 4,
                                        "padding_h": 3, "padding_w": 3, "dilation_h": 1, "dilation_w": 1, "groups": 1,
                                        "padding_mode": "zeros"},
                  "time_per_run_ms": [29.92336082458496, 30.14041519165039, 29.685792922973633, 30.10665512084961,
                                      30.054399490356445, 29.895647048950195, 29.76464080810547, 29.944799423217773,
                                      29.9182071685791, 30.67184066772461, 30.1527042388916, 29.9366397857666,
                                      30.625823974609375, 31.971391677856445, 29.950944900512695, 30.061567306518555,
                                      30.079008102416992, 29.808639526367188, 29.922271728515625, 29.928447723388672,
                                      30.05846405029297, 30.059551239013672, 29.897727966308594, 30.171167373657227,
                                      30.00320053100586, 30.16089630126953, 29.884416580200195, 29.853696823120117,
                                      29.92742347717285, 29.592575073242188, 29.677600860595703, 30.075904846191406,
                                      29.739967346191406, 29.666303634643555, 29.733888626098633, 29.837312698364258,
                                      29.442047119140625, 29.559839248657227, 29.660127639770508, 29.455360412597656,
                                      29.56598472595215, 29.84342384338379, 29.488128662109375, 29.70217514038086,
                                      29.447168350219727, 29.789215087890625, 29.74006462097168, 29.269023895263672,
                                      29.85273551940918, 29.872127532958984]}
        record = {"runtime_ms": 4154.592625, "iterations": 50, "valid_iterations": 50, "metric": "time",
                  "dvfs_config": {"cpu_freq": 576000, "gpu_freq": 1377000000, "memory_freq": 1600000000},
                  "kernel_parameters": {"n": 1, "c": 1168, "h": 280, "w": 280, "in_channels": 1168, "out_channels": 392,
                                        "kernel_h": 1, "kernel_w": 1, "bias": 1, "stride_h": 1, "stride_w": 1,
                                        "padding_h": 2, "padding_w": 2, "dilation_h": 1, "dilation_w": 1, "groups": 1,
                                        "padding_mode": "zeros"},
                  "time_per_run_ms": [80.72978973388672, 80.4801254272461, 80.26787567138672, 80.15462493896484,
                                      80.21094512939453, 80.01878356933594, 80.48979187011719, 80.91686248779297,
                                      80.34591674804688, 80.73260498046875, 80.33817291259766, 80.58844757080078,
                                      79.99075317382812, 81.04934692382812, 81.2748794555664, 81.84268951416016,
                                      80.81884765625, 80.43609619140625, 80.34467315673828, 80.62921905517578,
                                      80.3971176147461, 80.61996459960938, 80.3703384399414, 80.29798126220703,
                                      80.7919692993164, 80.46403503417969, 80.26214599609375, 80.46249389648438,
                                      80.4032974243164, 80.31436920166016, 80.58943939208984, 80.2213134765625,
                                      80.50025939941406, 80.30448150634766, 80.31324768066406, 80.49801635742188,
                                      80.48332977294922, 80.89299011230469, 80.70121765136719, 80.52531433105469,
                                      80.33010864257812, 80.89997100830078, 80.49449920654297, 80.3752670288086,
                                      80.31715393066406, 80.11571502685547, 80.31846618652344, 80.32768249511719,
                                      80.25759887695312, 80.439453125]}
        record = {"runtime_ms": 2739.553622, "iterations": 50, "valid_iterations": 50, "metric": "time",
                  "dvfs_config": {"cpu_freq": 1344000, "gpu_freq": 1032750000, "memory_freq": 1065600000},
                  "kernel_parameters": {"n": 1, "c": 1280, "h": 154, "w": 245, "num_features": 1280},
                  "time_per_run_ms": [8.275808334350586, 8.28003215789795, 8.286208152770996, 8.283103942871094,
                                      8.28320026397705, 8.269824028015137, 8.269023895263672, 8.273599624633789,
                                      8.282143592834473, 8.283967971801758, 8.278016090393066, 8.270784378051758,
                                      8.269280433654785, 8.275839805603027, 8.272928237915039, 8.283167839050293,
                                      8.275168418884277, 8.272895812988281, 8.276512145996094, 8.280096054077148,
                                      8.276991844177246, 8.27507209777832, 8.277664184570312, 8.274592399597168,
                                      8.273216247558594, 8.277024269104004, 8.279007911682129, 8.281087875366211,
                                      8.274592399597168, 8.273568153381348, 8.275936126708984, 8.268383979797363,
                                      8.270336151123047, 8.277567863464355, 8.277152061462402, 8.275712013244629,
                                      8.276224136352539, 8.278911590576172, 8.274144172668457, 8.269536018371582,
                                      8.271136283874512, 8.274975776672363, 8.279680252075195, 8.270848274230957,
                                      8.273632049560547, 8.284704208374023, 8.276960372924805, 8.27683162689209,
                                      8.278656005859375, 8.266752243041992]}
        record = {"runtime_ms": 2161.61197, "iterations": 50, "valid_iterations": 50, "metric": "time",
                  "dvfs_config": {"cpu_freq": 2112000, "gpu_freq": 675750000, "memory_freq": 2133000000},
                  "kernel_parameters": {"n": 1, "c": 688, "h": 189, "w": 28, "num_features": 688},
                  "time_per_run_ms": [2.9757440090179443, 2.970207929611206, 2.973695993423462, 2.96726393699646,
                                      2.969599962234497, 2.972640037536621, 2.9754879474639893, 2.9797439575195312,
                                      2.9641919136047363, 2.9675519466400146, 2.972223997116089, 2.968575954437256,
                                      2.973695993423462, 2.970367908477783, 2.970367908477783, 2.9674880504608154,
                                      2.9655039310455322, 2.973695993423462, 2.9681599140167236, 2.9666879177093506,
                                      2.9767680168151855, 2.968672037124634, 2.9622719287872314, 2.9663360118865967,
                                      2.9721920490264893, 2.964479923248291, 2.9590399265289307, 2.964479923248291,
                                      2.9614078998565674, 2.973695993423462, 2.963360071182251, 2.973695993423462,
                                      2.9714879989624023, 2.9675519466400146, 2.9665279388427734, 2.9706239700317383,
                                      2.98470401763916, 2.9948160648345947, 2.9665279388427734, 2.964672088623047,
                                      2.9685120582580566, 2.9675838947296143, 2.965536117553711, 2.9767680168151855,
                                      2.9640960693359375, 2.9686079025268555, 2.9624640941619873, 2.9712960720062256,
                                      2.9767680168151855, 2.973344087600708]}
        cpu_freq = record[S_DVFS_CONFIG][S_CPU_FREQ]
        gpu_freq = record[S_DVFS_CONFIG][S_GPU_FREQ]
        memory_freq = record[S_DVFS_CONFIG][S_MEMORY_FREQ]
        print(np.max(record[S_TIME_PER_RUN_MS]), np.min(record[S_TIME_PER_RUN_MS]))
        tmp = {
            S_CPU_FREQ: cpu_freq,
            S_GPU_FREQ: gpu_freq,
            S_MEMORY_FREQ: memory_freq,
            S_RUNTIME_MS: np.mean(record[S_TIME_PER_RUN_MS])
        }
        kernel_params = pd.DataFrame(record[S_KERNEL_PARAMETERS], index=[0])

        kernel = 'batchnorm2d'
        model = BatchNormModel()
        kernel_params = model.network_specific_preprocess_data(kernel_params).to_dict(orient='records')[0]

        input_record = {**kernel_params, **tmp}

        df = pd.DataFrame(input_record, index=[0])
        # print(df)
        # return
        df_summary = pd.DataFrame()
        df_summary[S_RUNTIME_MS] = df[S_RUNTIME_MS]

        model_inputs_df = df.drop(columns=[i for i in df.columns if i in ALL_LABELS])

        model_inputs_dm = xgb.DMatrix(model_inputs_df)
        model = ModelLoader.load_trained_model(kernel, S_RUNTIME_MS)
        prediction = model.predict(model_inputs_dm)
        print(prediction)
        df_summary[f'prediction'] = Transformers.inverse_log_transform(prediction)
        print(df_summary)

    @staticmethod
    def find_intersection_between_modeling_and_characterization():

        kernel = 'matmuladd'
        modeling_nodes = [f'node-0{i}' for i in range(1, 10)]
        char_nodes = [f'node-{i}' for i in range(11, 14)]

        model = TrainingHelpers.load_kernel_model(kernel)
        if model is None:
            return

        print("Kernel: ", kernel)
        df_char_dict = model.read_characterization_data_summary(nodes=char_nodes,
                                                                keep_labels=[S_AVG_PWR_W, S_RUNTIME_MS])
        df_modeling_dict = model.read_modeling_data_summary(nodes=modeling_nodes,
                                                            keep_labels=[S_AVG_PWR_W, S_RUNTIME_MS])
        non_label_columns = [i for i in list(df_char_dict.values())[0].columns if i not in ALL_LABELS]

        intersection_dict = dict()
        print(non_label_columns)
        for i in modeling_nodes:
            for j in char_nodes:
                # Merge method returns the other columns post fixed with x and y
                intersection_dict[f"{i}_{j}"] = \
                    Helpers.get_intersections(df_modeling_dict[i], df_char_dict[j], non_label_columns)
                print(f"{i}_{j}")
                print(intersection_dict[f"{i}_{j}"])
        # print("Number of intersecting records stats: ")
        # for key, val in intersection_dict.items():
        #     print(f"{key} : {val.shape[0]}")
        return intersection_dict


class Generator:
    @staticmethod
    def generate_config_file_randomly(kernel, num_records):
        config_file_path = join_path(CROSS_NODE_ANALYSIS, 'configurations', f"{kernel}_{num_records}.json")
        generator = KernelsLoaders.load_generator(kernel)

        seen_before = dict()
        kernel_parameters = generator.generate_random_input_parameters()
        dvfs_config = DVFSHelpers.generate_random_dvfs_config()
        seen_before[str(dvfs_config)] = kernel_parameters.to_id()
        item = {S_DVFS_CONFIG: dvfs_config, S_KERNEL_PARAMETERS: kernel_parameters.to_dict()}
        FileUtils.serialize(item, config_file_path, file_extension='json', append=True)

        for i in range(num_records - 1):
            dvfs_config = DVFSHelpers.generate_random_dvfs_config()
            kernel_parameters = generator.generate_random_input_parameters()

            while seen_before.get(str(dvfs_config), -1) == kernel_parameters.to_id():
                dvfs_config = DVFSHelpers.generate_random_dvfs_config()
                kernel_parameters = generator.generate_random_input_parameters()

            seen_before[str(dvfs_config)] = kernel_parameters.to_id()

            dvfs_config = DVFSHelpers.generate_random_dvfs_config()
            item = {S_DVFS_CONFIG: dvfs_config, S_KERNEL_PARAMETERS: kernel_parameters.to_dict()}
            FileUtils.serialize(item, config_file_path, file_extension='json', append=True)

    @staticmethod
    def check_duplicates(fp):
        # fp = '/home/ubuntu/Projects/jetson-modeling/generated/cross-node-analysis/configurations/conv2d_1000.json'
        with open(fp, 'r') as fo:
            lines = fo.readlines()
            x = list()
            for line in lines:
                record = json.loads(line)
                if record not in x:
                    x.append(record)
                else:
                    print(record)
            # print(len(x), len(lines))
            return len(x), len(lines)

    @staticmethod
    def generate_config_file_from_modeling_configs(kernel, num_records, modeling_nodes):
        output_config_file_path = join_path(CROSS_NODE_ANALYSIS, 'configurations', f"{kernel}_{num_records}.json")

        kernel_model = TrainingHelpers.load_kernel_model(kernel)
        configurations = kernel_model.parse_configurations_from_modeling_data(nodes=modeling_nodes)

        seen_before = set()
        kernel_parameters_parser = KernelsLoaders.load_kernel_parameters_parser(kernel)

        picked_config = pick(configurations)
        dvfs_config = picked_config[S_DVFS_CONFIG]
        kernel_parameters = kernel_parameters_parser(picked_config[S_KERNEL_PARAMETERS])
        lookup_key = DVFSHelpers.get_dvfs_config_id(dvfs_config) + '_' + kernel_parameters.to_csv(delimiter='_')
        seen_before.add(lookup_key)
        item = {S_DVFS_CONFIG: dvfs_config, S_KERNEL_PARAMETERS: kernel_parameters.to_dict()}
        FileUtils.serialize(item, output_config_file_path, file_extension='json', append=True)

        for _ in range(num_records - 1):
            picked_config = pick(configurations)
            dvfs_config = picked_config[S_DVFS_CONFIG]
            kernel_parameters = kernel_parameters_parser(picked_config[S_KERNEL_PARAMETERS])
            lookup_key = DVFSHelpers.get_dvfs_config_id(dvfs_config) + '_' + kernel_parameters.to_csv(delimiter='_')
            while seen_before.__contains__(lookup_key):
                picked_config = pick(configurations)
                dvfs_config = picked_config[S_DVFS_CONFIG]
                kernel_parameters = kernel_parameters_parser(picked_config[S_KERNEL_PARAMETERS])
                lookup_key = DVFSHelpers.get_dvfs_config_id(dvfs_config) + '_' + kernel_parameters.to_csv(delimiter='_')

            seen_before.add(lookup_key)

            item = {S_DVFS_CONFIG: dvfs_config, S_KERNEL_PARAMETERS: kernel_parameters.to_dict()}
            FileUtils.serialize(item, output_config_file_path, file_extension='json', append=True)

    @staticmethod
    def generate_random_dvfs_configs_file(num_records):
        seen_before = set()
        for _ in range(num_records):
            config = DVFSHelpers.generate_random_dvfs_config_partial()
            while seen_before.__contains__(DVFSHelpers.get_dvfs_config_id(config)):
                config = DVFSHelpers.generate_random_dvfs_config_partial()

            seen_before.add(DVFSHelpers.get_dvfs_config_id(config))
            FileUtils.serialize(config, join_path(CROSS_NODE_ANALYSIS_CONFIGURATIONS, f'dvfs_{num_records}'),
                                append=True, file_extension='json')

    @staticmethod
    def entry(num_records):
        kernels = ['batchnorm2d', 'matmul', 'add', 'relu', 'cat']
        nodes = ['node-%02d' % i for i in range(1, 14)]
        print(nodes)
        for kernel in kernels:
            Generator.generate_config_file_from_modeling_configs(kernel, num_records, nodes)
            # Generator.generate_config_file_randomly(kernel, num_records)


def main():
    nodes = ['node-%02d' % i for i in range(1, 14)]
    kernels = ['conv2d']
    for kernel in kernels:
        print(f"Analysis of : {kernel}")
        intersection_dict = Helpers.find_records_with_common_parameters(kernel, nodes)
        Helpers.report_stats(kernel, intersection_dict)


def simulate_training():
    nodes = ['node-%02d' % i for i in range(1, 14)]
    kernels = ['conv2d']
    for kernel in kernels:
        print(f"Analysis of : {kernel}")
        Helpers.simulate_ideal_model_training(kernel)


def test():
    Helpers.manual_model_inspection()
    # Helpers.find_intersection_between_modeling_and_characterization()
    # Generator.generate_random_dvfs_configs_file(525)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        if sys.argv[1] == 'main':
            main()
        elif sys.argv[1] == 'generate':
            Generator.entry(int(sys.argv[2]))
        elif sys.argv[1] == 'simulate':
            simulate_training()
        elif sys.argv[1] == 'reload':
            if len(sys.argv) == 3:
                Helpers.reload_results_and_report_stats(sys.argv[2])
            else:
                Helpers.reload_results_and_report_stats()
        else:
            pass
    else:
        test()
        # main()
