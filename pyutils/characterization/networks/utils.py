import os.path

from pyutils.common.paths import VISION_NETWORKS_PROFILING_OUTPUT_DIR

_VISION_NETWORKS = ["resnet", "densenet", "vgg", "googlenet", "shufflenet", "mobilenet", "mnasnet", "alexnet",
                    "squeezenet", "inception"]


def get_unified_benchmark_name(bm: str):
    if bm is None:
        return bm
    bm = bm.lower()
    if bm in ['resnet']:
        return "resnet18"
    if bm in ['alexnet']:
        return "alexnet"
    if bm in ['vgg']:
        return "vgg16"
    if bm in ['googlenet']:
        return "googlenet"
    if bm in ['squeezenet']:
        return "squeezenet1_0"
    if bm in ['mnasnet']:
        return "mnasnet1_0"
    if bm in ['densenet']:
        return "densenet161"
    elif bm in ['shufflenetv2', 'shufflenet']:
        return 'shufflenet_v2_x1_0'
    elif bm in ['mobilenet', 'mobilenetv2']:
        return 'mobilenet_v2'
    elif bm in ['inception', 'inceptionv3', 'inception3']:
        return 'inception_v3'
    else:
        return bm


VISION_NETWORKS = [get_unified_benchmark_name(i) for i in sorted(_VISION_NETWORKS)]


def is_network(name: str):
    return get_unified_benchmark_name(name) in VISION_NETWORKS


def get_print_benchmark_name(nw_name):
    nw = get_unified_benchmark_name(nw_name)
    if nw in ['resnet18', 'alexnet', 'squeezenet1_0', 'mnasnet1_0', 'densenet161',
              "shufflenet_v2_x1_0", "mobilenet_v2", "inception_v3"]:
        nw = nw.capitalize()
        nw = nw.replace('net', 'Net')
        return nw
    elif nw == 'googlenet':
        return 'GoogeLeNet'
    elif nw == 'vgg16':
        return 'VGG16'
    elif is_network(nw):
        nw = nw.replace('net', 'Net')
        return nw
    else:
        return nw.capitalize()


class NetworksNameHelpers:
    @staticmethod
    def get_supported_networks():
        return [get_unified_benchmark_name(i) for i in VISION_NETWORKS]

    @staticmethod
    def is_googlenet(nw):
        return get_unified_benchmark_name(nw) == "googlenet"

    @staticmethod
    def is_inception(nw):
        return get_unified_benchmark_name(nw) == "inception_v3"


def get_nw_stack_trace_file_path(nw, version="v2"):
    nw = get_unified_benchmark_name(nw)
    # V2: uses JP5 and Pytorch >=1.11.
    if version == "v2":
        return os.path.join(VISION_NETWORKS_PROFILING_OUTPUT_DIR, nw, f"{nw}_stack_v2.txt")
    else:
        return os.path.join(VISION_NETWORKS_PROFILING_OUTPUT_DIR, nw, f"stack_content.txt")


def get_nw_last_executed_graph_file_path(nw, version="v2"):
    nw = get_unified_benchmark_name(nw)
    # V2: uses JP5 and Pytorch >=1.11.
    if version == "v2":
        return os.path.join(VISION_NETWORKS_PROFILING_OUTPUT_DIR, nw, f"{nw}_last_executed_graph.txt")
    else:
        raise FileNotFoundError
