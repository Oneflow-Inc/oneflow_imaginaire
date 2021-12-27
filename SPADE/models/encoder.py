import oneflow as flow


def conv_encoder(input, trainable=True):
    crop_size = 512
    if input.shape[2]!=256 or input.shape[3]!=256:
        input = flow.experimental.nn.UpsamplingBilinear2d(size=(256, 256))(input)

    input = flow.layers.conv2d(input, 64, kernel_size=3, strides=2, padding=1, trainable=trainable)
    input = flow.nn.InstanceNorm2d(input, affine=trainable)

    input = flow.nn.leaky_relu(input, 2e-1)
    input = flow.layers.conv2d(input, 64*2, kernel_size=3, strides=2, padding=1, trainable=trainable)
    input = flow.nn.InstanceNorm2d(input, affine=trainable)

    input = flow.nn.leaky_relu(input, 2e-1)
    input = flow.layers.conv2d(input, 64 * 4, kernel_size=3, strides=2, padding=1, trainable=trainable)
    input = flow.nn.InstanceNorm2d(input, affine=trainable)

    input = flow.nn.leaky_relu(input, 2e-1)
    input = flow.layers.conv2d(input, 64 * 8, kernel_size=3, strides=2, padding=1, trainable=trainable)
    input = flow.nn.InstanceNorm2d(input, affine=trainable)

    input = flow.nn.leaky_relu(input, 2e-1)
    input = flow.layers.conv2d(input, 64 * 8, kernel_size=3, strides=2, padding=1, trainable=trainable)
    input = flow.nn.InstanceNorm2d(input, affine=trainable)

    if crop_size>256:
        input = flow.nn.leaky_relu(input, 2e-1)
        input = flow.layers.conv2d(input, 64 * 8, kernel_size=3, strides=2, padding=1, trainable=trainable)
        input = flow.nn.InstanceNorm2d(input, affine=trainable)

    input = flow.nn.leaky_relu(input, 2e-1)

    input = flow.reshape(input, [input.shape[0], -1])

    mu = flow.layers.dense(input, 256, trainable=trainable)
    logvar = flow.layers.dense(input, 256, trainable=trainable)

    return mu, logvar
