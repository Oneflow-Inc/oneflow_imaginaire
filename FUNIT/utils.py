from typing import Callable, Dict
import numpy as np
import oneflow as flow


def namer_factory() -> Callable[[str], str]:
    record = {} # type: Dict[str, int]

    def namer(op_name: str) -> str:
        if op_name in record:
            record[op_name] += 1
        else:
            # count from 1
            record[op_name] = 1

        index = record[op_name]

        return f"{op_name}{index}"

    return namer


def conv2d_layer(
    input, out_channel, kernel_size=3, strides=1, padding="SAME", 
    data_format="NCHW", dilation_rate=1, use_bias=True, 
    weight_initializer=flow.random_normal_initializer(mean=0.0, stddev=0.02), 
    bias_initializer=flow.zeros_initializer(), name="conv2d"
): 

    weight_shape = (out_channel, input.shape[1], kernel_size, kernel_size)

    with flow.scope.namespace(name):
        weight = flow.get_variable(
            "weight", 
            shape=weight_shape, dtype=input.dtype, 
            initializer=weight_initializer
        )

    output = flow.nn.conv2d(
        input, weight, strides, padding, data_format, dilation_rate)

    if use_bias:
        with flow.scope.namespace(name):
            bias = flow.get_variable(
                "bias", 
                shape=(out_channel,), dtype=input.dtype, 
                initializer=bias_initializer
            )

        output = flow.nn.bias_add(output, bias, data_format)

    return output


def dense_layer(
    input, units, use_bias=True, 
    weight_initializer=flow.zeros_initializer(),
    bias_initializer=flow.zeros_initializer(), name="dense"
):

    weight_shape = (units, input.shape[1])

    with flow.scope.namespace(name):
        weight = flow.get_variable(
            "weight", 
            shape=weight_shape, dtype=input.dtype, 
            initializer=weight_initializer
        )

    out = flow.matmul(input, weight, transpose_b=True)

    if use_bias:
        with flow.scope.namespace(name):
            bias = flow.get_variable(
                "bias", 
                shape=(units,), dtype=input.dtype, 
                initializer=bias_initializer
            )

        out = flow.nn.bias_add(out, bias)
    
    return out


def embedding_layer(indices, num_embeddings, embedding_dim, name="embedding"):
    weight_shape = (num_embeddings, embedding_dim)

    with flow.scope.namespace(name):
        weight = flow.get_variable(
            "weight", 
            shape=weight_shape, dtype=flow.float32, 
            initializer=flow.random_normal_initializer()
        )

        out = flow.gather(params=weight, indices=indices, axis=0)

    return out
