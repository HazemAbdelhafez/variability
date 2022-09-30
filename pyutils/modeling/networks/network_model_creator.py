import argparse

from pyutils.common.config import *
from pyutils.common.strings import S_METRIC
from pyutils.common.utils import GlobalLogger
from pyutils.modeling.helpers import TrainingHelpers

target_label = S_RUNTIME_MS
logger = GlobalLogger().get_logger()


def entry(args: dict):
    max_num_nodes = (13 if args['num_nodes'] is None else int(args['num_nodes']))
    nodes = ['node-%02d' % i for i in range(1, max_num_nodes + 1)]

    overwrite_data_summary = (False if args['overwrite'] is None
                              else True if str(args['overwrite']).lower() == 'true' else False)
    test_nodes_id = ('2,13' if args['test_nodes'] is None else args['test_nodes'])

    _test_nodes = ['node-%02d' % int(i) for i in test_nodes_id.split(',')]
    _train_nodes = [node for node in nodes if node not in _test_nodes]
    prefix = (None if args['prefix'] is None else str(args['prefix']))

    if args['single_node'] is not None:
        _train_nodes = ['node-%02d' % int(args['single_node'])]

    if str(args['networks']).__contains__(','):
        networks = str(args['networks']).split(',')
    else:
        networks = [str(args['networks'])]

    label = TrainingHelpers.get_label(args[S_METRIC])
    logger.info(f"Selected networks:   {networks}")
    logger.info(f"Selected metric:     {args['metric']} -> Label: {label}")
    logger.info(f"Nodes:               {1} to {max_num_nodes}")
    logger.info(f"Train nodes:         "
                f"{','.join([str(i) for i in range(1, max_num_nodes + 1) if str(i) not in test_nodes_id.split(',')])}")
    logger.info(f"Single node:         {args['single_node']}")
    logger.info(f"Test nodes:          {test_nodes_id}")

    logger.info(f"Modeling label:      {target_label}")

    # Do we report the training and testing data sizes only or not
    report_stats_only = (False if args['stats_only'] is None
                         else True if str(args['stats_only']).lower() == 'true' else False)
    if report_stats_only:
        logger.info("Reporting status only.")

    for network in networks:
        # Load a training model for the target network
        model = TrainingHelpers.load_network_model(network)

        # Create (train and optimize) the networks model
        model.create_model(_train_nodes, _test_nodes, label, overwrite_data_summary, stats_only=report_stats_only,
                           prefix=prefix)


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("-k", "--networks", required=True)
    argument_parser.add_argument("-m", "--metric", required=True)
    argument_parser.add_argument("-tn", "--test_nodes", required=False)
    argument_parser.add_argument("-o", "--overwrite", required=False)
    argument_parser.add_argument("-n", "--num_nodes", required=False)
    argument_parser.add_argument("-t", "--timestamp", required=False)
    argument_parser.add_argument("-s", "--stats_only", required=False)
    argument_parser.add_argument("-sn", "--single_node", required=False)
    argument_parser.add_argument("-p", "--prefix", required=False)
    cmd_args = vars(argument_parser.parse_args())
    target_label = cmd_args['metric']
    entry(cmd_args)
