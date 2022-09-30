import argparse

from pyutils.modeling.helpers import TrainingHelpers

DEFAULT_TEST_NODES = ['2', '13']


def entry(args):
    # Parse cmd args
    kernel = args['kernel']
    label = args['metric']
    reload = (True if str(args['reload']).lower() == 'true' else False)
    overwrite = (False if str(args['overwrite']).lower() == 'false' else True)
    model = TrainingHelpers.load_kernel_model(kernel)

    max_num_nodes = (13 if args['num_nodes'] is None else int(args['num_nodes']))
    nodes = ['node-%02d' % i for i in range(1, max_num_nodes + 1)]
    test_nodes_id = (','.join(DEFAULT_TEST_NODES) if args['test_nodes'] is None else args['test_nodes'])
    test_nodes = ['node-%02d' % int(i) for i in test_nodes_id.split(',')]
    train_nodes = [node for node in nodes if node not in test_nodes]

    if not reload:
        print("[ANALYSIS DATA FORMATTER] Retraining model")

        best_params = {"objective": "reg:squarederror", "eval_metric": ["rmse"], "tree_method": "hist",
                       "colsample_bytree": 0.9, "gamma": 0, "learning_rate": 0.1, "max_bin": 512, "max_depth": 5,
                       "min_child_weight": 7, "reg_alpha": 0.001, "reg_lambda": 0.001, "subsample": 0.8}

        model.train_and_evaluate_model_with_tuned_parameters(training_nodes=train_nodes,
                                                             testing_nodes=test_nodes,
                                                             best_params=best_params,
                                                             best_num_boost_round=None,
                                                             target_label=label, overwrite_data_summary=overwrite)
    else:
        timestamp = args['timestamp']
        print("[ANALYSIS DATA FORMATTER] Loading trained model")
        meta_file_path = model.re_evaluate_for_reporting(timestamp, load_model=False)
        print(f"[ANALYSIS DATA FORMATTER] Re-evaluated model meta-file path: {meta_file_path}")


if __name__ == '__main__':
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("-k", "--kernel", required=True)
    argument_parser.add_argument("-m", "--metric", required=True)
    argument_parser.add_argument("-tn", "--test_nodes", required=True)
    argument_parser.add_argument("-n", "--num_nodes", required=True)
    argument_parser.add_argument("-t", "--timestamp", required=False)
    argument_parser.add_argument("-r", "--reload", required=True)
    argument_parser.add_argument("-o", "--overwrite", required=False)
    cmd_args = vars(argument_parser.parse_args())
    entry(cmd_args)
