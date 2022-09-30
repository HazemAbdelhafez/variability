import torch
from pyutils.characterization.kernels.conv2d_parameters_analysis import Conv2dGenerator, Conv2dParameters

from pyutils.common.timers import CudaStopWatch

with torch.jit.optimized_execution(True):
    with torch.no_grad():
        # kernel_params = Conv2dGenerator.generate_random_input_parameters()
        kernel_params = {'n': 1, 'c': 32, 'h': 28, 'w': 28, 'in_channels': 32, 'out_channels': 192, 'kernel_h': 1,
                         'kernel_w': 1, 'bias': 0, 'stride_h': 1, 'stride_w': 1, 'padding_h': 0, 'padding_w': 0,
                         'dilation_h': 1, 'dilation_w': 1, 'groups': 1, 'padding_mode': 'zeros', 'generator_version': 3}
        # print(kernel_params.to_dict())
        kernel_params = Conv2dParameters.from_dict(kernel_params)
        kernel_input = Conv2dGenerator.create_input(kernel_params)
        kernel = Conv2dGenerator.generate_module(kernel_params)

        # warmup
        # Uses static_input and static_target here for convenience,
        # but in a real setting, because the warmup includes optimizer.step()
        # you must use a few batches of real data.
        # s = torch.cuda.Stream()
        # s.wait_stream(torch.cuda.current_stream())
        # with torch.cuda.stream(s):
        for i in range(5):
            y_pred = kernel(kernel_input)
        # torch.cuda.current_stream().wait_stream(s)
        # print(kernel_input)
        # print(kernel)
        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            timer = CudaStopWatch()
            timer.start()
            for _ in range(10):
                # Fills the graph's input memory with new data to compute on
                # static_input.copy_(data)
                # replay() includes forward, backward, and step.
                # You don't even need to call optimizer.zero_grad() between iterations
                # because the captured backward refills static .grad tensors in place.
                y_pred = kernel(kernel_input)
                # Params have been updated. static_y_pred, static_loss, and .grad
                # attributes hold values from computing on this iteration's data.
            timer.stop()
            print(timer.elapsed_ms(10))
