import pytest
import numpy as np
import torch
import slangpy as spy
import slangpy_nn as nn

from slang_bench import device, linear_eval_module
from slang_bench.linear_layer import LinearLayer

BATCH_SIZES = [1024, 2048, 4096, 8192]
LINEAR_LAYER_SIZES = [(16, 64), (64, 64), (256, 256)]


@pytest.mark.parametrize("thread_count", BATCH_SIZES)
@pytest.mark.parametrize("num_inputs, num_outputs", LINEAR_LAYER_SIZES)
def test_linear_layer_slang(benchmark, thread_count, num_inputs, num_outputs):
    # Create reference solution using PyTorch.
    torch.manual_seed(42)
    input_tensor = torch.randn(thread_count, num_inputs)
    linear_layer_torch = torch.nn.Linear(num_inputs, num_outputs)
    linear_layer_torch.weight.data.uniform_(-1.0, 1.0)
    linear_layer_torch.bias.data.uniform_(-1.0, 1.0)
    with torch.no_grad():
        output_tensor = linear_layer_torch(input_tensor)

    # Create input buffer.
    input_buf = spy.NDBuffer(
        device=device,
        dtype=linear_eval_module.find_struct(f"float[{num_inputs}]"),
        shape=(thread_count,),
    )
    input_buf.copy_from_numpy(input_tensor.numpy().astype(np.float32))
    # Create output buffer.
    output_buf = spy.NDBuffer(
        device=device,
        dtype=linear_eval_module.find_struct(f"float[{num_outputs}]"),
        shape=(thread_count,),
    )
    # Create a linear layer with random weights and biases.
    linear_layer = LinearLayer(num_inputs=num_inputs, num_outputs=num_outputs)
    # Copy the weights and biases from the PyTorch model to the Slang model.
    linear_layer.weight.copy_from_numpy(
        linear_layer_torch.weight.data.numpy().astype(np.float32)
    )
    linear_layer.bias.copy_from_numpy(
        linear_layer_torch.bias.data.numpy().astype(np.float32)
    )

    # Run the benchmark.
    def run():
        eval_fn = linear_eval_module.find_function(f"eval<{num_outputs}, {num_inputs}>")
        assert (
            eval_fn is not None
        ), f"eval function for LinearLayer<{num_inputs}, {num_outputs}> not found"
        eval_fn(
            model=linear_layer.get_this(),
            input=input_buf,
            _result=output_buf,
        )
        device.wait()
        # Get the output from the output buffer and convert it to a NumPy array.
        output_array = output_buf.to_numpy().reshape(thread_count, num_outputs)
        # Compare the output with the reference solution.
        np.testing.assert_allclose(
            output_array, output_tensor.numpy(), rtol=1e-5, atol=1e-5
        )

    benchmark(run)


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("num_inputs, num_outputs", LINEAR_LAYER_SIZES)
def test_linear_layer_torch(benchmark, batch_size, num_inputs, num_outputs):
    torch.manual_seed(42)
    input_tensor = torch.randn(batch_size, num_inputs)
    linear_layer = torch.nn.Linear(num_inputs, num_outputs)
    linear_layer.weight.data.uniform_(-1.0, 1.0)
    linear_layer.bias.data.uniform_(-1.0, 1.0)

    def run():
        output_tensor = linear_layer(input_tensor)

    benchmark(run)


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("num_inputs, num_outputs", LINEAR_LAYER_SIZES)
def test_linear_layer_backward_torch(benchmark, batch_size, num_inputs, num_outputs):
    torch.manual_seed(42)
    input_tensor = torch.randn(batch_size, num_inputs)
    linear_layer = torch.nn.Linear(num_inputs, num_outputs)
    linear_layer.weight.data.uniform_(-1.0, 1.0)
    linear_layer.bias.data.uniform_(-1.0, 1.0)

    def run():
        linear_layer.zero_grad()
        # Check that the gradients of the weights and biases are None before the backward pass.
        assert linear_layer.weight.grad is None
        assert linear_layer.bias.grad is None
        output_tensor = linear_layer(input_tensor)
        output_tensor.sum().backward()
        # Check that the gradients of the weights and biases are not None.
        assert linear_layer.weight.grad is not None
        assert linear_layer.bias.grad is not None

    benchmark(run)
