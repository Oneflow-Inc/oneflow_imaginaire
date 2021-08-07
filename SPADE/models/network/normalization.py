import oneflow as flow
import cv2
import numpy as np

def conv2d_layer(
    input, num_filters, kernel_size=3, strides=1, padding="SAME",
    data_format="NCHW", dilation_rate=1, use_bias=True,
    weight_initializer=flow.xavier_uniform_initializer(),
    bias_initializer=flow.zeros_initializer(), name="conv2d",
    trainable=None
):

    weight_shape = (num_filters, input.shape[1], kernel_size, kernel_size)

    with flow.scope.namespace(name):
        weight = flow.get_variable(
            "weight",
            shape=weight_shape,
            dtype=input.dtype,
            initializer=weight_initializer,
            trainable=trainable
        )

    output = flow.nn.conv2d(
        input, weight, strides, padding, data_format, dilation_rate)

    if use_bias:
        with flow.scope.namespace(name):
            bias = flow.get_variable(
                "bias",
                shape=(num_filters,),
                dtype=input.dtype,
                initializer=bias_initializer,
                trainable=trainable
            )

        output = flow.nn.bias_add(output, bias, data_format)

    return output

def deconv(input, out_channel, name_prefix, kernel_size = 4, strides = [2, 2], trainable = True, reuse = True):
    weight = flow.get_variable(
        name_prefix + "_weight",
        shape = (input.shape[1], out_channel, kernel_size, kernel_size),
        dtype = flow.float,
        initializer = flow.xavier_uniform_initializer(),
        trainable = trainable,
        reuse = reuse
    )
    return flow.nn.conv2d_transpose(
                input,
                weight,
                strides = strides,
                padding = "SAME",
                output_shape = (input.shape[0], out_channel, input.shape[2] * strides[0], input.shape[3] * strides[1]))

# Input produce batchnorm activation
# segmap produce scale and bias
def spade(input, segmap, ks=3, pf_norm='batch', trainable=True, name_prefix='spade'):
    """

    @param config:
    @param norm_nc: the #channels of the normalized activations, hence the output dim of SPADE
    @param segmap:
    @param input:
    @param ks: the size of kernel in the SPADE module (e.g. 3x3)
    @param pf_norm: the type of parameter-free normalization. (e.g. syncbatch, batch, instance)
    """
    # the number of input.shape[1] equals to the number of norm_nc?

    norm_nc = input.shape[1]
    with flow.scope.namespace(name_prefix):
        def mlp_shared(segmap_interpolate, nhidden, ks, trainable=trainable):
            out = conv2d_layer(segmap_interpolate, nhidden, kernel_size=ks, trainable=trainable, padding='SAME', name=name_prefix+'mlp_shared')
            out = flow.nn.relu(out)
            return out

        if pf_norm == 'batch':
            param_free_norm = flow.layers.batch_normalization(input, axis=1, trainable=trainable, name='pf_norm', center=False, scale=False)
        else:
            raise ('Other batch methods No implement!')

        actv = mlp_shared(segmap, 128, ks, trainable=trainable)
        gamma = conv2d_layer(actv, norm_nc, ks, trainable=trainable, padding='SAME', name='gamma')
        beta = conv2d_layer(actv, norm_nc, ks, trainable=trainable, padding='SAME', name='beta')
        out = param_free_norm * (1+gamma) + beta

    return out

def spadeRes(input, segmap, out_c, spectral=True, trainable=True, name_prefix='spadeRes'):
    learned_shortcut = (input.shape[1] != out_c) # bug
    middle_c = min(input.shape[1], out_c)

    if spectral==True:
        raise ('Have not implement spectral norm')

    def shortcut(x, seg, trainable=True):
        if learned_shortcut:
            if spectral==True:
                raise ('Have not implement spectral norm')
            else:
                with flow.scope.namespace(name_prefix):
                    x_s = conv2d_layer(spade(x, seg), out_c, kernel_size=1, trainable=trainable, use_bias=False,name='shortcut')
                # there is a relu between conv2d and spade in paper, but disappear in NVlas.
        else:
            x_s = x
        return x_s

    def actvn(x):
        return flow.nn.leaky_relu(x, alpha=2e-1)
    with flow.scope.namespace(name_prefix):
        x_s = shortcut(input, segmap, trainable=trainable)

        x = spade(input, segmap, trainable=trainable, name_prefix='spadeRes_spade1')
        x = actvn(x)
        x = conv2d_layer(x, middle_c, trainable=trainable, name='spadeRes_conv2d_1')
        x = spade(x, segmap, trainable=trainable, name_prefix='spadeRes_spade2')
        x = actvn(x)
        x = conv2d_layer(x, out_c, trainable=trainable, name='spadeRes_conv2d_2')

    if learned_shortcut:
        return x+x_s
    else:
        return x
