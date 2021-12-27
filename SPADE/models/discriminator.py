import oneflow as flow


def multi_scale_discriminator(input, trainable=True):
    num_d = 2
    get_intermediate_featuers = True

    results = []
    for i in range(num_d):
        with flow.scope.namespace('multi_scale_discriminator'+str(i)):
            out = n_layer_discriminator(input, trainable=trainable)
            if not get_intermediate_featuers:
                out = [out]
            results.append(out)
            input = flow.nn.avg_pool2d(input, ksize=3, strides=2, padding='SAME')
    return results



def n_layer_discriminator(input, trainable=True):
    init = flow.xavier_uniform_initializer()
    reg = flow.regularizers
    nf = 64
    n_layer_d = 4
    get_intermediate_featuers = True
    results = [input]

    out = flow.layers.conv2d(input, nf, kernel_size=4, strides=2, padding='SAME', trainable=trainable, name='D_n_1', kernel_initializer=init)
    out = flow.nn.leaky_relu(out, 2e-1)
    results.append(out)

    nf = min(nf * 2, 512)
    out = flow.layers.conv2d(out, nf, kernel_size=4, strides=2, padding='SAME', trainable=trainable, name='D_n_2', kernel_initializer=init)
    out = flow.nn.InstanceNorm2d(out, name='D_n_I_1', affine=False)
    out = flow.nn.leaky_relu(out, 2e-1)
    results.append(out)

    nf = min(nf * 2, 512)
    out = flow.layers.conv2d(out, nf, kernel_size=4, strides=2, padding='SAME', trainable=trainable, name='D_n_3', kernel_initializer=init)
    out = flow.nn.InstanceNorm2d(out, name='D_n_I_2', affine=False)
    out = flow.nn.leaky_relu(out, 2e-1)
    results.append(out)

    nf = min(nf * 2, 512)
    out = flow.layers.conv2d(out, nf, kernel_size=4, strides=1, padding='SAME', trainable=trainable, name='D_n_4', kernel_initializer=init)
    out = flow.nn.InstanceNorm2d(out, name='D_n_I_3', affine=False)
    out = flow.nn.leaky_relu(out, 2e-1)
    results.append(out)

    out = flow.layers.conv2d(out, 1, kernel_size=4, strides=1, padding='SAME', trainable=trainable, name='D_n_5', kernel_initializer=init)
    results.append(out)

    if get_intermediate_featuers:
        return results[1:]
    else:
        return results[-1]