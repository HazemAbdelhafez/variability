from os.path import join as jp
from typing import List

from pyutils.common.arguments import RodiniaBlockBasedMeasurePowerArgs, RodiniaBlockBasedMeasureTimeArgs
from pyutils.common.paths import BENCHMARKS_DIR
from pyutils.common.strings import S_ITERATIONS, S_WARMUP_ITERATIONS, S_RUNTIME_MODE, S_TIMING_METHOD, \
    S_NUM_OBSERVATIONS, \
    S_BLOCK_RUNTIME_MS, S_BLOCK_SIZE, S_OUTPUT_DATA_FILE, S_RME, S_CONFIDENCE_LVL
from pyutils.common.utils import GlobalLogger
from pyutils.run.config import S_BM_INPUT_VERSION, NN
from pyutils.run.config import S_KMEANS_INPUT_VERSION, S_THREADS, S_MAT_DIM, S_NW_PENALTY, S_NW_MAX_DIM, \
    S_HUFFMAN_TIMING_METHOD, S_HUFFMAN_INPUT_VERSION, S3D as s3d
from pyutils.run.utils import PathsHandler

logger = GlobalLogger().get_logger()


class BenchmarkBase:
    benchmark_id = 'None'
    bm_exe = jp(BENCHMARKS_DIR, benchmark_id)

    def __init__(self, args):
        # self.cm_args contains benchmark specific arguments (e.g., mat dim)
        args = args.replace(' ', '')
        splits = args.split(',')
        bm_cmd_args = {}
        for s in splits:
            if s == '' or s == ' ':
                continue
            key, value = s.split('=')
            bm_cmd_args[key] = value
        self.cmd_args = bm_cmd_args
        self.runtime_mode = self.cmd_args[S_RUNTIME_MODE]
        self.num_observations = -1

    def to_csv(self, delimiter='_') -> str:
        return f'{delimiter}'.join([i for i in self.to_dict().values()])

    def to_dict(self) -> dict:
        raise NotImplemented

    def get_general_args(self, kwargs: dict):
        if isinstance(kwargs, RodiniaBlockBasedMeasurePowerArgs):
            default_args = ['-f', kwargs.output_data_file, '-e', '1', '-r', kwargs.num_observations,
                            '-b', kwargs.block_size, '-u', kwargs.block_runtime_ms, '-m', kwargs.rme,
                            '-c', kwargs.confidence_lvl, '-w', kwargs.warmup_itrs]
        elif isinstance(kwargs, RodiniaBlockBasedMeasureTimeArgs):
            default_args = ['-f', kwargs.output_data_file, '-e', '0', '-r', kwargs.num_observations,
                            '-b', kwargs.block_size, '-u', kwargs.block_runtime_ms, '-m', kwargs.rme,
                            '-c', kwargs.confidence_lvl, '-w', kwargs.warmup_itrs]
        else:
            # TODO: will be obsolete soon. Remove it.
            default_args = ['-f', kwargs[S_OUTPUT_DATA_FILE], '-e', self.runtime_mode, '-r', kwargs[S_NUM_OBSERVATIONS],
                            '-b', kwargs[S_BLOCK_SIZE], '-u', kwargs[S_BLOCK_RUNTIME_MS], '-m', kwargs[S_RME],
                            '-c', kwargs[S_CONFIDENCE_LVL], '-w', kwargs[S_WARMUP_ITERATIONS]]
        return default_args

    def get_run_cmd(self, kwargs: dict) -> List[str]:
        # kwargs: contains generic arguments (e.g., num of observations)
        raise NotImplemented


class Hotspot3D(BenchmarkBase):
    benchmark_id = 'hotspot3d'
    bm_exe = jp(BENCHMARKS_DIR, benchmark_id)

    def __init__(self, args):
        super().__init__(args)
        self.dim = self.cmd_args[s3d.dim]
        self.layers = self.cmd_args[s3d.layers]
        self.p_file = self.cmd_args[s3d.p_file]
        self.t_file = self.cmd_args[s3d.t_file]

    def to_dict(self) -> dict:
        tmp = dict()
        tmp[s3d.dim] = self.dim
        tmp[s3d.layers] = self.layers
        tmp[s3d.p_file] = self.p_file
        tmp[s3d.t_file] = self.t_file
        tmp[S_RUNTIME_MODE] = self.runtime_mode
        return tmp

    def get_run_cmd(self, kwargs: dict) -> List[str]:
        temp_file = jp(BENCHMARKS_DIR, f"temp_{self.t_file}")
        power_file = jp(BENCHMARKS_DIR, f"power_{self.p_file}")
        cmd = [self.bm_exe, '-d', self.dim, '-l', self.layers, '-p', power_file, '-t', temp_file]

        cmd = cmd + self.get_general_args(kwargs)
        cmd = [str(i) for i in cmd]
        return cmd


class Hotspot3DOpenMP(Hotspot3D):
    benchmark_id = 'hotspot3d_omp'
    bm_exe = jp(BENCHMARKS_DIR, benchmark_id)

    def __init__(self, args):
        super().__init__(args)
        self.threads = self.cmd_args[S_THREADS]

    def to_dict(self) -> dict:
        tmp = super().to_dict()
        tmp.update({S_THREADS: self.threads})
        return tmp

    def get_run_cmd(self, kwargs: dict) -> List[str]:
        temp_file = jp(BENCHMARKS_DIR, f"temp_{self.t_file}")
        power_file = jp(BENCHMARKS_DIR, f"power_{self.p_file}")
        cmd = [self.bm_exe, '-d', self.dim, '-l', self.layers, '-p', power_file, '-t', temp_file, '-q', self.threads]

        cmd = cmd + self.get_general_args(kwargs)
        cmd = [str(i) for i in cmd]
        return cmd


class Huffman(BenchmarkBase):
    benchmark_id = 'huffman'
    bm_exe = jp(BENCHMARKS_DIR, benchmark_id)

    def __init__(self, args):
        super().__init__(args)
        self.timing_method = self.cmd_args[S_HUFFMAN_TIMING_METHOD]
        self.input_version = self.cmd_args[S_HUFFMAN_INPUT_VERSION]

    def to_csv(self, delimiter='_'):
        return f'{self.timing_method}_{self.input_version}'

    def to_dict(self):
        return {S_HUFFMAN_TIMING_METHOD: self.timing_method,
                S_HUFFMAN_INPUT_VERSION: self.input_version}

    def get_run_cmd(self, kwargs: dict):
        input_data_file_path = jp(BENCHMARKS_DIR, f"huffman_input_{self.input_version}.in")
        cmd = [self.bm_exe, '-i', input_data_file_path]

        cmd = cmd + self.get_general_args(kwargs)
        cmd = [str(i) for i in cmd]
        return cmd


class Needle(BenchmarkBase):
    benchmark_id = 'nw'
    bm_exe = jp(BENCHMARKS_DIR, benchmark_id)

    def __init__(self, args):
        super().__init__(args)
        self.max_dim = self.cmd_args[S_NW_MAX_DIM]
        self.penalty = self.cmd_args[S_NW_PENALTY]

    def to_csv(self, delimiter='_'):
        return f'{self.max_dim}_{self.penalty}'

    def to_dict(self):
        return {S_NW_MAX_DIM: self.max_dim, S_NW_PENALTY: self.penalty}

    def get_run_cmd(self, kwargs: dict):
        cmd = [self.bm_exe, '-d', self.max_dim, '-p', self.penalty]
        cmd = cmd + self.get_general_args(kwargs)
        cmd = [str(i) for i in cmd]
        return cmd


class NearestNeighbor(BenchmarkBase):
    benchmark_id = 'nn'
    bm_exe = jp(BENCHMARKS_DIR, benchmark_id)

    def __init__(self, args):
        super().__init__(args)
        self.input_file = self.cmd_args[NN.input_file]
        self.results = self.cmd_args[NN.results]
        self.lng = self.cmd_args[NN.lng]
        self.lat = self.cmd_args[NN.lat]

    def to_csv(self, delimiter='_'):
        return f'{self.input_file}_{self.results}_{self.lng}_{self.lat}'

    def to_dict(self):
        return {NN.input_file: self.input_file, NN.results: self.results, NN.lng: self.lng, NN.lat: self.lat}

    def get_run_cmd(self, kwargs: dict):
        input_files_list = jp(BENCHMARKS_DIR, 'nn_data', self.input_file)
        cmd = [self.bm_exe, '-i', input_files_list, '-n', self.results, '-t', self.lat, '-g', self.lng]
        cmd = cmd + self.get_general_args(kwargs)
        cmd = [str(i) for i in cmd]
        return cmd


class LUDCUDAParams(BenchmarkBase):
    benchmark_id = 'lud'
    bm_exe = jp(BENCHMARKS_DIR, benchmark_id)

    def __init__(self, args):
        super().__init__(args)
        self.mat_dim = self.cmd_args[S_MAT_DIM]

    def to_csv(self, delimiter='_'):
        return f'{self.mat_dim}'

    def to_dict(self):
        return {S_MAT_DIM: self.mat_dim}

    def get_run_cmd(self, kwargs: dict):
        # cmd = [self.bm_exe, '-s', self.mat_dim, kwargs[S_ITERATIONS], self.runtime_mode, self.num_pwr_observations]
        cmd = [self.bm_exe, '-s', self.mat_dim]

        cmd = cmd + self.get_general_args(kwargs)
        cmd = [str(i) for i in cmd]
        return cmd


class LUDOPENMPParams(LUDCUDAParams):
    benchmark_id = 'lud_omp'
    bm_exe = jp(BENCHMARKS_DIR, benchmark_id)

    def __init__(self, args):
        super().__init__(args)
        self.threads = self.cmd_args[S_THREADS]

    def to_csv(self, delimiter='_'):
        return f'{self.mat_dim}_{self.threads}'

    def to_dict(self):
        return {S_MAT_DIM: self.mat_dim, S_THREADS: self.threads}

    def get_run_cmd(self, kwargs: dict):
        cmd = [self.bm_exe, '-s', self.mat_dim, '-q', self.threads]

        cmd = cmd + self.get_general_args(kwargs)
        cmd = [str(i) for i in cmd]
        return cmd


class KMeansCUDA(BenchmarkBase):
    benchmark_id = 'kmeans_cuda'
    bm_exe = jp(BENCHMARKS_DIR, benchmark_id)

    def __init__(self, args):
        super().__init__(args)
        self.input_version = self.cmd_args[S_KMEANS_INPUT_VERSION]

    def to_csv(self, delimiter='_'):
        return f'{self.input_version}'

    def to_dict(self):
        return {S_KMEANS_INPUT_VERSION: self.input_version}

    def get_run_cmd(self, kwargs: dict):
        input_data_file_path = jp(BENCHMARKS_DIR, f"kmeans_cuda_input_{self.input_version}")
        cmd = [self.bm_exe, '-i', str(input_data_file_path), kwargs[S_ITERATIONS], self.runtime_mode,
               self.num_observations]
        cmd = [str(i) for i in cmd]
        return cmd


class BFS(BenchmarkBase):
    benchmark_id = 'bfs'
    bm_exe = jp(BENCHMARKS_DIR, benchmark_id)

    def __init__(self, args):
        super().__init__(args)
        self.input_version = self.cmd_args[S_BM_INPUT_VERSION]

    def to_csv(self, delimiter='_'):
        return f'{self.input_version}'

    def to_dict(self):
        return {S_BM_INPUT_VERSION: self.input_version}

    def get_run_cmd(self, kwargs: dict):
        input_data_file_path = jp(BENCHMARKS_DIR, f"bfs_input_{self.input_version}")
        cmd = [self.bm_exe, '-i', input_data_file_path]
        cmd = cmd + self.get_general_args(kwargs)
        cmd = [str(i) for i in cmd]
        return cmd


class Sequences(BenchmarkBase):
    benchmark_id = 'torch_sequences'
    bm_exe = ''

    def __init__(self, args):
        super().__init__(args)
        self.timing_method = self.cmd_args[S_TIMING_METHOD]
        self.seq = self.cmd_args['seq']

    def to_csv(self, delimiter='_'):
        return f'{self.seq}_{self.timing_method}'

    def to_dict(self) -> dict:
        tmp = dict()
        tmp['seq'] = self.seq
        tmp[S_TIMING_METHOD] = self.timing_method
        return tmp

    def get_run_cmd(self, kwargs: dict) -> List[str]:
        py_exe = PathsHandler.get_python_remote_path()
        cmd = [py_exe, '-m', 'pyutils.run.run_sequences',
               '-s', self.seq,
               '-t', self.timing_method,
               '-i', kwargs[S_ITERATIONS]
               ]

        cmd = [str(i) for i in cmd]
        return cmd
