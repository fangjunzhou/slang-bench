import pytest
import logging
import numpy as np
import torch
import slangpy as spy
import slangpy_nn as nn

from slang_bench import device, linear_eval_module
from slang_bench.linear_layer import LinearLayer, LinearLayerAtomic

torch_device = torch.device("cpu")
if torch.cuda.is_available():
    torch_device = torch.device("cuda")
elif torch.backends.mps.is_available():
    torch_device = torch.device("mps")


logger = logging.getLogger(__name__)


BATCH_SIZES = [1024, 2048, 4096, 8192]
LINEAR_LAYER_SIZES = [(16, 64), (64, 64), (128, 128), (256, 256)]


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

    eval_fn = linear_eval_module.find_function(f"eval<{num_outputs}, {num_inputs}>")
    assert (
        eval_fn is not None
    ), f"eval function for LinearLayer<{num_inputs}, {num_outputs}> not found"

    eval_fn(
        model=linear_layer.get_this(),
        input=input_buf,
        _result=output_buf,
    )
    # Get the output from the output buffer and convert it to a NumPy array.
    output_array = output_buf.to_numpy().reshape(thread_count, num_outputs)
    # Compare the output with the reference solution.
    np.testing.assert_allclose(
        output_array, output_tensor.numpy(), rtol=1e-5, atol=1e-5
    )

    # Run the benchmark.
    def run():
        eval_fn(
            model=linear_layer.get_this(),
            input=input_buf,
            _result=output_buf,
        )
        device.wait()

    benchmark.pedantic(run, rounds=100)


@pytest.mark.parametrize("thread_count", BATCH_SIZES)
@pytest.mark.parametrize("num_inputs, num_outputs", LINEAR_LAYER_SIZES)
def test_linear_layer_backward_slang(benchmark, thread_count, num_inputs, num_outputs):
    # Create reference solution using PyTorch.
    torch.manual_seed(42)
    input_tensor = torch.randn(thread_count, num_inputs)
    linear_layer_torch = torch.nn.Linear(num_inputs, num_outputs)
    linear_layer_torch.weight.data.uniform_(-1.0, 1.0)
    linear_layer_torch.bias.data.uniform_(-1.0, 1.0)

    # Forward pass.
    output_tensor = linear_layer_torch(input_tensor)
    output_tensor.retain_grad()
    s = output_tensor.sum()
    s.backward()
    # Get the linear layer gradients from PyTorch.
    weight_grad_torch = linear_layer_torch.weight.grad
    assert weight_grad_torch is not None
    weight_grad_np = weight_grad_torch.numpy()
    bias_grad_torch = linear_layer_torch.bias.grad
    assert bias_grad_torch is not None
    bias_grad_np = bias_grad_torch.numpy()

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
    # Create output gradient buffer.
    output_grad_buf = spy.NDBuffer(
        device=device,
        dtype=linear_eval_module.find_struct(f"float[{num_outputs}]"),
        shape=(thread_count,),
    )
    output_grad_buf.copy_from_numpy(output_tensor.grad.numpy().astype(np.float32))
    # Run the backward function for the linear layer.
    backward_fn = linear_eval_module.find_function(
        f"evalBackward<{num_outputs}, {num_inputs}>"
    )
    assert (
        backward_fn is not None
    ), f"evalBackward function for LinearLayer<{num_inputs}, {num_outputs}> not found"
    backward_fn(
        model=linear_layer.get_this(),
        input=input_buf,
        gradOutput=output_grad_buf,
    )
    # Get the weight and bias gradients from the linear layer.
    weight_grad_array = linear_layer.weight_grad.to_numpy().reshape(
        num_outputs, num_inputs
    )
    bias_grad_array = linear_layer.bias_grad.to_numpy().reshape(num_outputs)
    # Compare the weight and bias gradients with the reference solution.
    # We expect the gradient to be different because the gradient without atomic adds is not exactly the same as the PyTorch gradient.
    weight_diff = weight_grad_array - weight_grad_np
    bias_diff = bias_grad_array - bias_grad_np
    # Show the first 8 elements of the weight and bias gradients and their differences.
    logger.info(
        f"Weight grad (first 8 elements): {weight_grad_array.flatten()[:8]}"
        f"\nWeight grad PyTorch (first 8 elements): {weight_grad_np.flatten()[:8]}"
        f"\nWeight grad diff (first 8 elements): {weight_diff.flatten()[:8]}"
        f"\nBias grad (first 8 elements): {bias_grad_array.flatten()[:8]}"
        f"\nBias grad PyTorch (first 8 elements): {bias_grad_np.flatten()[:8]}"
        f"\nBias grad diff (first 8 elements): {bias_diff.flatten()[:8]}"
    )

    def run():
        backward_fn(
            model=linear_layer.get_this(),
            input=input_buf,
            gradOutput=output_grad_buf,
        )
        device.wait()

    benchmark.pedantic(run, rounds=100)


@pytest.mark.parametrize("thread_count", BATCH_SIZES)
@pytest.mark.parametrize("num_inputs, num_outputs", LINEAR_LAYER_SIZES)
def test_linear_layer_atomic_slang(benchmark, thread_count, num_inputs, num_outputs):
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
    linear_layer = LinearLayerAtomic(num_inputs=num_inputs, num_outputs=num_outputs)
    # Copy the weights and biases from the PyTorch model to the Slang model.
    linear_layer.weight.copy_from_numpy(
        linear_layer_torch.weight.data.numpy().astype(np.float32)
    )
    linear_layer.bias.copy_from_numpy(
        linear_layer_torch.bias.data.numpy().astype(np.float32)
    )

    eval_fn = linear_eval_module.find_function(
        f"evalAtomic<{num_outputs}, {num_inputs}>"
    )
    assert (
        eval_fn is not None
    ), f"evalAtomic function for LinearLayerAtomic<{num_inputs}, {num_outputs}> not found"

    eval_fn(
        model=linear_layer.get_this(),
        input=input_buf,
        _result=output_buf,
    )
    # Get the output from the output buffer and convert it to a NumPy array.
    output_array = output_buf.to_numpy().reshape(thread_count, num_outputs)
    # Compare the output with the reference solution.
    np.testing.assert_allclose(
        output_array, output_tensor.numpy(), rtol=1e-5, atol=1e-5
    )

    # Run the benchmark.
    def run():
        eval_fn(
            model=linear_layer.get_this(),
            input=input_buf,
            _result=output_buf,
        )
        device.wait()

    benchmark.pedantic(run, rounds=100)


@pytest.mark.parametrize("thread_count", BATCH_SIZES)
@pytest.mark.parametrize("num_inputs, num_outputs", LINEAR_LAYER_SIZES)
def test_linear_layer_backward_atomic_slang(
    benchmark, thread_count, num_inputs, num_outputs
):
    # Create reference solution using PyTorch.
    torch.manual_seed(42)
    input_tensor = torch.randn(thread_count, num_inputs)
    linear_layer_torch = torch.nn.Linear(num_inputs, num_outputs)
    linear_layer_torch.weight.data.uniform_(-1.0, 1.0)
    linear_layer_torch.bias.data.uniform_(-1.0, 1.0)

    # Forward pass.
    output_tensor = linear_layer_torch(input_tensor)
    output_tensor.retain_grad()
    s = output_tensor.sum()
    s.backward()
    # Get the linear layer gradients from PyTorch.
    weight_grad_torch = linear_layer_torch.weight.grad
    assert weight_grad_torch is not None
    weight_grad_np = weight_grad_torch.numpy()
    bias_grad_torch = linear_layer_torch.bias.grad
    assert bias_grad_torch is not None
    bias_grad_np = bias_grad_torch.numpy()

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
    linear_layer = LinearLayerAtomic(num_inputs=num_inputs, num_outputs=num_outputs)
    # Copy the weights and biases from the PyTorch model to the Slang model.
    linear_layer.weight.copy_from_numpy(
        linear_layer_torch.weight.data.numpy().astype(np.float32)
    )
    linear_layer.bias.copy_from_numpy(
        linear_layer_torch.bias.data.numpy().astype(np.float32)
    )
    # Create output gradient buffer.
    output_grad_buf = spy.NDBuffer(
        device=device,
        dtype=linear_eval_module.find_struct(f"float[{num_outputs}]"),
        shape=(thread_count,),
    )
    output_grad_buf.copy_from_numpy(output_tensor.grad.numpy().astype(np.float32))
    # Run the backward function for the linear layer.
    backward_fn = linear_eval_module.find_function(
        f"evalBackwardAtomic<{num_outputs}, {num_inputs}>"
    )
    assert (
        backward_fn is not None
    ), f"evalBackwardAtomic function for LinearLayerAtomic<{num_inputs}, {num_outputs}> not found"
    backward_fn(
        model=linear_layer.get_this(),
        input=input_buf,
        gradOutput=output_grad_buf,
    )
    # Get the weight and bias gradients from the linear layer.
    weight_grad_array = linear_layer.weight_grad.to_numpy().reshape(
        num_outputs, num_inputs
    )
    bias_grad_array = linear_layer.bias_grad.to_numpy().reshape(num_outputs)
    # Compare the weight and bias gradients with the reference solution.
    # We expect the gradient to be the same as the PyTorch gradient when using atomic adds.
    weight_diff = weight_grad_array - weight_grad_np
    bias_diff = bias_grad_array - bias_grad_np
    # Show the first 8 elements of the weight and bias gradients and their differences.
    logger.info(
        f"Weight grad (first 8 elements): {weight_grad_array.flatten()[:8]}"
        f"\nWeight grad PyTorch (first 8 elements): {weight_grad_np.flatten()[:8]}"
        f"\nWeight grad diff (first 8 elements): {weight_diff.flatten()[:8]}"
        f"\nBias grad (first 8 elements): {bias_grad_array.flatten()[:8]}"
        f"\nBias grad PyTorch (first 8 elements): {bias_grad_np.flatten()[:8]}"
        f"\nBias grad diff (first 8 elements): {bias_diff.flatten()[:8]}"
    )
    assert np.allclose(
        weight_grad_array, weight_grad_np, rtol=1e-3, atol=1e-3
    ), "Weight gradients should be close to PyTorch gradients when using atomic adds"
    assert np.allclose(
        bias_grad_array, bias_grad_np, rtol=1e-3, atol=1e-3
    ), "Bias gradients should be close to PyTorch gradients when using atomic adds"

    def setup():
        # Zero out the gradient buffers before each benchmark iteration
        linear_layer.weight_grad.copy_from_numpy(
            np.zeros(num_outputs * num_inputs, dtype=np.float32)
        )
        linear_layer.bias_grad.copy_from_numpy(np.zeros(num_outputs, dtype=np.float32))

    def run():
        backward_fn(
            model=linear_layer.get_this(),
            input=input_buf,
            gradOutput=output_grad_buf,
        )
        device.wait()

    benchmark.pedantic(run, setup=setup, rounds=100)


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

    benchmark.pedantic(run, rounds=100)


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("num_inputs, num_outputs", LINEAR_LAYER_SIZES)
def test_linear_layer_backward_torch(benchmark, batch_size, num_inputs, num_outputs):
    torch.manual_seed(42)
    input_tensor = torch.randn(batch_size, num_inputs)
    linear_layer = torch.nn.Linear(num_inputs, num_outputs)
    linear_layer.weight.data.uniform_(-1.0, 1.0)
    linear_layer.bias.data.uniform_(-1.0, 1.0)

    # Forward pass.
    output_tensor = linear_layer(input_tensor)
    output_tensor.retain_grad()
    # Compute a simple loss (sum of outputs) and perform backward pass to compute gradients.
    s = output_tensor.sum()
    s.backward(retain_graph=True)
    # Get the output tensor gradients.
    output_grad = output_tensor.grad
    output_tensor.grad = None
    assert output_grad is not None and not np.allclose(
        output_grad.numpy(), 0.0
    ), "Output gradients should not be zero after backward"
    # Zero the gradients.
    linear_layer.zero_grad()
    # Check that the gradients are zero after calling zero_grad.
    assert linear_layer.weight.grad is None or np.allclose(
        linear_layer.weight.grad.numpy(), 0.0
    ), "Weight gradients should be zero after zero_grad"
    assert linear_layer.bias.grad is None or np.allclose(
        linear_layer.bias.grad.numpy(), 0.0
    ), "Bias gradients should be zero after zero_grad"
    # Backward on linear layer.
    output_tensor.backward(output_grad, retain_graph=True)
    # Check that the gradients are not None after backward pass.
    assert linear_layer.weight.grad is not None and not np.allclose(
        linear_layer.weight.grad.numpy(), 0.0
    ), "Weight gradients should not be zero after backward"
    assert linear_layer.bias.grad is not None and not np.allclose(
        linear_layer.bias.grad.numpy(), 0.0
    ), "Bias gradients should not be zero after backward"

    def setup():
        linear_layer.zero_grad()

    def run():
        output_tensor.backward(output_grad, retain_graph=True)

    benchmark.pedantic(run, setup=setup, rounds=100)


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("num_inputs, num_outputs", LINEAR_LAYER_SIZES)
def test_linear_layer_torch_gpu(benchmark, batch_size, num_inputs, num_outputs):
    torch.manual_seed(42)
    input_tensor = torch.randn(batch_size, num_inputs).to(torch_device)
    linear_layer = torch.nn.Linear(num_inputs, num_outputs).to(torch_device)
    linear_layer.weight.data.uniform_(-1.0, 1.0)
    linear_layer.bias.data.uniform_(-1.0, 1.0)

    def run():
        output_tensor = linear_layer(input_tensor)
        torch.mps.synchronize()

    benchmark.pedantic(run, rounds=100)


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("num_inputs, num_outputs", LINEAR_LAYER_SIZES)
def test_linear_layer_backward_torch_gpu(
    benchmark, batch_size, num_inputs, num_outputs
):
    torch.manual_seed(42)
    input_tensor = torch.randn(batch_size, num_inputs).to(torch_device)
    linear_layer = torch.nn.Linear(num_inputs, num_outputs).to(torch_device)
    linear_layer.weight.data.uniform_(-1.0, 1.0)
    linear_layer.bias.data.uniform_(-1.0, 1.0)

    # Forward pass.
    output_tensor = linear_layer(input_tensor)
    output_tensor.retain_grad()
    # Compute a simple loss (sum of outputs) and perform backward pass to compute gradients.
    s = output_tensor.sum()
    s.backward(retain_graph=True)
    # Get the output tensor gradients.
    output_grad = output_tensor.grad
    output_tensor.grad = None
    assert output_grad is not None and not np.allclose(
        output_grad.cpu().numpy(), 0.0
    ), "Output gradients should not be zero after backward"
    # Zero the gradients.
    linear_layer.zero_grad()
    # Check that the gradients are zero after calling zero_grad.
    assert linear_layer.weight.grad is None or np.allclose(
        linear_layer.weight.grad.cpu().numpy(), 0.0
    ), "Weight gradients should be zero after zero_grad"
    assert linear_layer.bias.grad is None or np.allclose(
        linear_layer.bias.grad.cpu().numpy(), 0.0
    ), "Bias gradients should be zero after zero_grad"
    # Backward on linear layer.
    output_tensor.backward(output_grad, retain_graph=True)
    # Check that the gradients are not None after backward pass.
    assert linear_layer.weight.grad is not None and not np.allclose(
        linear_layer.weight.grad.cpu().numpy(), 0.0
    ), "Weight gradients should not be zero after backward"
    assert linear_layer.bias.grad is not None and not np.allclose(
        linear_layer.bias.grad.cpu().numpy(), 0.0
    ), "Bias gradients should not be zero after backward"

    def setup():
        linear_layer.zero_grad()

    def run():
        output_tensor.backward(output_grad, retain_graph=True)
        torch.mps.synchronize()

    benchmark.pedantic(run, setup=setup, rounds=100)
