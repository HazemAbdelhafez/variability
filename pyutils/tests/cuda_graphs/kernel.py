import torch

from pyutils.characterization.kernels.matmul_parameters_analysis import MatMulAddGenerator

with torch.jit.optimized_execution(True):
    with torch.no_grad():
        generator = MatMulAddGenerator()
        kernel_params = MatMulAddGenerator.generate_random_input_parameters()
        kernel_input = generator.create_input(kernel_params)
        kernel = generator.generate_module(kernel_params)

        # warmup
        # Uses static_input and static_target here for convenience,
        # but in a real setting, because the warmup includes optimizer.step()
        # you must use a few batches of real data.
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for i in range(5):
                y_pred = kernel(kernel_input)
        torch.cuda.current_stream().wait_stream(s)

        # capture
        g = torch.cuda.CUDAGraph()
        # Sets grads to None before capture, so backward() will create
        # .grad attributes with allocations from the graph's private pool
        with torch.cuda.graph(g):
            static_y_pred = kernel(kernel_input)

        for _ in range(20):
            # Fills the graph's input memory with new data to compute on
            # static_input.copy_(data)
            # replay() includes forward, backward, and step.
            # You don't even need to call optimizer.zero_grad() between iterations
            # because the captured backward refills static .grad tensors in place.
            g.replay()
            # Params have been updated. static_y_pred, static_loss, and .grad
            # attributes hold values from computing on this iteration's data.
