import numpy as np
import oneflow as flow
import utils

def StyleEncoder(
    input, num_downsamples, num_filters, style_channels, 
    padding_mode, activation_norm_type, weight_norm_type, 
    nonlinearity, name="StyleEncoder"
):

    assert padding_mode == "reflect"
    assert activation_norm_type == "none"
    assert weight_norm_type == ""
    assert nonlinearity == "relu"

    namer = utils.namer_factory()

    def Conv2dBlock(input, num_filters, kernel_size, strides, padding):
        out = flow.reflection_pad2d(input, padding=padding)
        out = utils.conv2d_layer(
            out, num_filters=num_filters, kernel_size=kernel_size, 
            strides=strides, padding="VALID", name=namer("conv")
        )
        out = flow.nn.relu(out)
        return out

    out = input
    with flow.scope.namespace(name):
        out = Conv2dBlock(
            out, num_filters=num_filters, 
            kernel_size=7, strides=1, padding=3
        )
        
        for _ in range(2):
            num_filters *= 2
            out = Conv2dBlock(
                out, num_filters=num_filters, 
                kernel_size=4, strides=2, padding=1
            )
        
        for _ in range(num_downsamples - 2):
            out = Conv2dBlock(
                out, num_filters=num_filters, 
                kernel_size=4, strides=2, padding=1
            )

        # AdaptiveAvgPool2D(1)
        out = flow.reshape(out, shape=[*out.shape[:2], 1, -1])
        out = flow.math.reduce_mean(out, axis=3, keepdims=True)

        out = utils.conv2d_layer(
            out, num_filters=style_channels, kernel_size=1, strides=1, 
            padding="VALID", use_bias=False, name=namer("conv")
        )

    return out


def ContentEncoder(
    input, num_downsamples, num_res_blocks, image_channels, num_filters,
    padding_mode, activation_norm_type, weight_norm_type, nonlinearity, 
    name="ContentEncoder"
):

    assert padding_mode == "reflect"
    assert activation_norm_type == "instance"
    assert weight_norm_type == ""
    assert nonlinearity == "relu"

    namer = utils.namer_factory()

    def Conv2dBlock(input, num_filters, kernel_size, strides, padding):
        out = input

        out = flow.reflection_pad2d(out, padding=padding)
        out = utils.conv2d_layer(
            out, num_filters=num_filters, kernel_size=kernel_size, 
            strides=strides, padding="VALID", name=namer("conv")
        )
        out = flow.nn.InstanceNorm2d(out, affine=True)
        out = flow.nn.relu(out)

        return out

    def Res2dBlock(input, num_filters):
        out = input

        for _ in range(2):
            out = flow.reflection_pad2d(out, padding=1)
            out = utils.conv2d_layer(
                out, num_filters=num_filters, kernel_size=3, 
                strides=1, padding="VALID", name=namer("conv")
            )
            out = flow.nn.InstanceNorm2d(out, affine=True)
            out = flow.nn.relu(out)

        if input.shape[1] == out.shape[1]:
            return input + out
        else:
            shortcut_out = utils.conv2d_layer(
                input, num_filters=num_filters, kernel_size=1, 
                strides=1, padding="VALID", name=namer("conv")
            )
            return shortcut_out + out

    out = input
    with flow.scope.namespace(name):
        out = Conv2dBlock(
            out, num_filters=num_filters, 
            kernel_size=7, strides=1, padding=3
        )

        for _ in range(num_downsamples):
            num_filters *= 2
            out = Conv2dBlock(
                out, num_filters=num_filters, 
                kernel_size=4, strides=2, padding=1
            )

        for _ in range(num_res_blocks):
            out = Res2dBlock(out, num_filters=num_filters)

    return out


def MLP(
    input, output_dim, latent_dim, num_layers,
    activation_norm_type, nonlinearity, name="MLP"
):

    assert activation_norm_type == "none"
    assert nonlinearity == "relu"
    assert len(input.shape) == 2

    namer = utils.namer_factory()

    def LinearBlock(input, out_features):
        out = utils.dense_layer(input, units=out_features, name=namer("dense"))
        out = flow.nn.relu(out)
        return out

    out = input
    with flow.scope.namespace(name):
        for _ in range(num_layers - 2):
            out = LinearBlock(out, out_features=latent_dim)

        out = LinearBlock(out, out_features=output_dim)
    
    return out


def Decoder(
    input, style, num_enc_output_channels, num_image_channels=3,
    num_upsamples=4, padding_type="reflect", weight_norm_type="",
    nonlinearity="relu", name="Decoder", 
    _persistent_namer=utils.namer_factory()
):

    assert padding_type == "reflect"
    assert weight_norm_type == ""
    assert nonlinearity == "relu"

    namer = utils.namer_factory()

    def AdaptiveNorm(input, style):
        assert len(input.shape) == 4
        assert len(style.shape) == 2

        out1 = flow.nn.InstanceNorm2d(input, affine=False)

        out2 = utils.dense_layer(
            style, units=input.shape[1] * 2, 
            use_bias=False, name=namer("dense"))

        gamma = flow.slice(out2, begin=[None, 0], size=[None, input.shape[1]])
        gamma = flow.reshape(gamma, shape=(*gamma.shape, 1, 1))

        beta = flow.slice(
            out2, begin=[None, input.shape[1]], size=[None, input.shape[1]]
        )
        beta = flow.reshape(gamma, shape=(*beta.shape, 1, 1))

        return out1 * (1 + gamma) + beta

    def base_res_block(input, style, num_filters):
        out = input

        for _ in range(2):
            out = flow.reflection_pad2d(out, padding=1)
            out = utils.conv2d_layer(
                out, num_filters=num_filters, kernel_size=3, 
                strides=1, padding="VALID", name=namer("conv")
            )
            out = AdaptiveNorm(out, style)
            out = flow.nn.relu(out)
        
        if input.shape[1] == out.shape[1]:
            return input + out
        else:
            shortcut_out = utils.conv2d_layer(
                input, num_filters=num_filters, kernel_size=1, 
                strides=1, padding="VALID", name=namer("conv")
            )
            return shortcut_out + out

    def base_up_res_block(input, style, num_filters):
        out = flow.reflection_pad2d(input, padding=2)
        out = utils.conv2d_layer(
            out, num_filters=num_filters, kernel_size=5, 
            strides=1, padding="VALID", name=namer("conv")
        )
        out = AdaptiveNorm(out, style)
        out = flow.nn.relu(out)

        out = flow.layers.upsample_2d(
            out, name=_persistent_namer("upsample")
        )

        out = flow.reflection_pad2d(out, padding=2)
        out = utils.conv2d_layer(
            out, num_filters=num_filters, kernel_size=5, 
            strides=1, padding="VALID", name=namer("conv")
        )
        out = AdaptiveNorm(out, style)
        out = flow.nn.relu(out)

        shortcut = flow.layers.upsample_2d(
            input, name=_persistent_namer("upsample")
        )
        
        if shortcut.shape[1] != out.shape[1]:
            shortcut = utils.conv2d_layer(
                shortcut, num_filters=num_filters, kernel_size=1, 
                strides=1, padding="VALID", name=namer("conv")
            )
            shortcut = flow.nn.InstanceNorm2d(shortcut, affine=True)

        return shortcut + out
    
    def Conv2dBlock(input, num_filters, kernel_size, strides, padding):
        out = input

        out = flow.reflection_pad2d(out, padding=padding)
        out = utils.conv2d_layer(
            out, num_filters=num_filters, kernel_size=kernel_size, 
            strides=strides, padding="VALID", name=namer("conv")
        )

        return flow.math.tanh(out)

    out = input
    with flow.scope.namespace(name):
        out = base_res_block(out, style, num_filters=num_enc_output_channels)
        out = base_res_block(out, style, num_filters=num_enc_output_channels)

        for _ in range(num_upsamples):
            num_enc_output_channels //= 2
            out = base_up_res_block(out, style, num_filters=num_enc_output_channels)

        out = Conv2dBlock(
            out, num_filters=num_image_channels, 
            kernel_size=7, strides=1, padding=3
        )

    return out
