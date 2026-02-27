import pathlib
import slangpy as spy
import slangpy_nn as nn

# The directory containing Slang shader files.
SHADERS_DIR = pathlib.Path(__file__).parent / "slang"

# Create a device that can be used across all benchmarks.
include_paths = nn.slang_include_paths() + [SHADERS_DIR]
device = spy.create_device(include_paths=include_paths)

# Load modules.
linear_layer_module = spy.Module.load_from_file(
    device=device, path="linear_layer.slang"
)
linear_eval_module = spy.Module.load_from_file(
    device=device, path="linear_eval.slang", link=[linear_layer_module]
)
