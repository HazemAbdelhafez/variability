EARLY_STOPPING_ROUNDS = 200
LOG_TRANSFORM = True
DEFAULT_NUM_BOOST_ROUNDS = 2000
DEFAULT_TUNING_ITERATIONS = 3000
MIN_BUDGET = 500
MAX_BUDGET = 2000

SUPPORTED_KERNELS = ["conv2d", "batchnorm2d", "maxpool2d", "adaptivepool", "cat", "add", "matmuladd", "relu",
                     "cudnn_convolution_add_relu", "cudnn_convolution_add"]
SUPPORTED_KERNELS_ALIASES = ["batch_norm", "max_pool2d", "adaptive_avg_pool2d", "cat", "add_", "mm", "relu_"]
ALL_SUPPORTED_KERNELS = SUPPORTED_KERNELS + SUPPORTED_KERNELS_ALIASES

UNSUPPORTED_KERNELS = ["hardtanh_", "unsqueeze", "transpose", "view", "reshape", "size", "dropout_", "flatten",
                       "chunk", "contiguous", "mul", "avg_pool2d", "dropout", "mean", "t",
                       "prim::ConstantChunk"]

UNHANDLED_KERNELS = ["_ncf_unsqueeze", "prim::BailoutTemplate", "prim::FusionGroup",
                     "dropout", "div", "Int", "prim::ConstantChunk", "slice", "select", "prim::TensorExprGroup"]
