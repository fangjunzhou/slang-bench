import slangpy as spy

from slang_bench import device, linear_eval_module


class LinearLayer:
    def __init__(self, num_inputs: int, num_outputs: int):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.weight = spy.NDBuffer(
            device=device,
            dtype=linear_eval_module.float,
            usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
            shape=(num_outputs * num_inputs,),
        )
        self.bias = spy.NDBuffer(
            device=device,
            dtype=linear_eval_module.float,
            usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
            shape=(num_outputs,),
        )
        self.weight_grad = spy.NDBuffer(
            device=device,
            dtype=linear_eval_module.float,
            usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
            shape=(num_outputs * num_inputs,),
        )
        self.bias_grad = spy.NDBuffer(
            device=device,
            dtype=linear_eval_module.float,
            usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
            shape=(num_outputs,),
        )

    def get_this(self):
        return {
            "weight": self.weight,
            "bias": self.bias,
            "weight_grad": self.weight_grad,
            "bias_grad": self.bias_grad,
            "_type": f"LinearLayer<{self.num_outputs}, {self.num_inputs}>",
        }


class LinearLayerAtomic:
    def __init__(self, num_inputs: int, num_outputs: int):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.weight = spy.NDBuffer(
            device=device,
            dtype=linear_eval_module.float,
            usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
            shape=(num_outputs * num_inputs,),
        )
        self.bias = spy.NDBuffer(
            device=device,
            dtype=linear_eval_module.float,
            usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
            shape=(num_outputs,),
        )
        self.weight_grad = spy.NDBuffer(
            device=device,
            dtype=linear_eval_module.find_struct("Atomic<float>[1]").as_struct(),
            usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
            shape=(num_outputs * num_inputs,),
        )
        self.bias_grad = spy.NDBuffer(
            device=device,
            dtype=linear_eval_module.find_struct("Atomic<float>[1]").as_struct(),
            usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
            shape=(num_outputs,),
        )

    def get_this(self):
        return {
            "weight": self.weight,
            "bias": self.bias,
            "weight_grad": self.weight_grad,
            "bias_grad": self.bias_grad,
            "_type": f"LinearLayerAtomic<{self.num_outputs}, {self.num_inputs}>",
        }
